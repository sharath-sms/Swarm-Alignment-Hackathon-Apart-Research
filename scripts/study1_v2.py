"""Study 1 v2 (EXPLORATORY; fixes external-critique issues M6, M9). No API/network calls.

A. Censored log-normal AFT ("Tobit") fit of time-to-tip on the Ashery et al. Claude N=24 archive:
     log T_i ~ Normal(a - gamma * log(p_i - p_c), sigma)     for p_i > p_c
   using ALL runs at cm >= cm_min; right-censored runs contribute log S = log Phi(-(log c - mu)/sigma).
   Models: CRIT (gamma, p_c free), CRIT_half (gamma = 1/2), CRIT_one (gamma = 1), DIL (p_c = 0, no threshold).
   The likelihood is profiled on a dense (p_c, gamma) grid (inner maximisation over a, sigma is a concave 2-parameter
   censored-normal problem solved by vectorised Newton in Olsen's (a/sigma, 1/sigma) parametrisation), then polished.
   Profile-likelihood 95% CIs (chi2_1). p_c is necessarily bounded above by the smallest p in the fit (cm_min / N).
   Choice for sub-threshold levels (cm = 2, 3): the PRIMARY fit uses cm >= 4 (all 112 runs incl. the censored cm=4 run).
   A SENSITIVITY fit ("mixture") uses all runs cm >= 2: runs with p > p_c follow the AFT law, runs with p <= p_c are
   modelled by a separate nuisance censored log-normal (noise-induced escape; 2 free params, counted for every p_c, so
   the profile over p_c is a proper likelihood). This lets the data decide whether cm=3 (p=0.125) lies above p_c
   (then its 9/12 censored runs must be explained by the power law) or below (escape process), and lifts the hard
   bound p_c < 4/24 to p_c < 4/24 with cm=3 in either branch. A third fit drops cm=4 (cm >= 5) to show leverage.
B. Parameter recovery at the Claude design (N=24, cm 2..12, same runs/cm, horizon 800 interactions, 2000 where the
   archive had 2000): (i) best-response emulator, (ii) DeepSeek pooled-policy emulator, (iii) synthetic log-normal AFT
   with known gamma in {0.5, 1.0}, p_c, a, sigma matched to the Claude fit under that gamma. Each dataset is fitted with
   the identical primary procedure. Plus the emulator's large-N, fine-grid "effective exponent".
   T50 in emulator runs uses the archive definition (first interaction at which the committed word reaches >= 50% of the
   last 2N non-committed answers): each round's non-committed answers are placed in random order across the round's
   N/2 interactions.
C. Kaplan-Meier median and RMST(800) per cm with bootstrap CIs (swarm_ews/survival.py).

Usage: .venv/bin/python scripts/study1_v2.py [--n-rec 300] [--workers 6] [--parts A,B,C,fig]
"""
from __future__ import annotations

import argparse, glob, json, pickle, re, sys, time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swarm_ews.emulator import best_response_policy, load_policy, simulate  # noqa: E402
from swarm_ews import survival  # noqa: E402

RAW = ROOT / "data/raw/ashery"
OUT = ROOT / "results/study1_time_to_tip"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "report/figures"; FIG.mkdir(parents=True, exist_ok=True)
HL2P = 0.5 * np.log(2 * np.pi)
CHI95 = chi2.ppf(0.95, 1)
N_ARCH, HORIZON = 24, 800


# ----------------------------------------------------------------------------------------------- data (safe unpickler)
class Safe(pickle.Unpickler):
    """Ashery pickles contain only builtin containers; block every class lookup (no code execution on load)."""
    def find_class(self, m, n):
        raise pickle.UnpicklingError(f"blocked {m}.{n}")


def t50_stream(ans, inter, N):
    """Archive T50: first interaction at which the rolling mean of the last W=2N non-committed answers >= 0.5."""
    W = 2 * N
    if len(ans) < W:
        return None
    roll = np.convolve(np.asarray(ans, float), np.ones(W) / W, mode="valid")
    hit = np.where(roll >= 0.5)[0]
    return int(inter[hit[0] + W - 1]) if len(hit) else None


def load_claude():
    rows = []
    for f in sorted(glob.glob(str(RAW / "claude_data/*swap*24ps*.pkl"))):
        m = re.search(r"swap_0_(\d+)cmtd_(\w)_(\w)_.*_(\d+)mem_.*_(\d+)ps", Path(f).name)
        cm, N = int(m[1]), int(m[5])
        with open(f, "rb") as fh:
            d = Safe(fh).load()
        for k, r in d.items():
            tr, word = r["tracker"], r["convergence"]["committed_to"]
            com = {pl for pl, v in r["simulation"].items() if v["committed_tag"]}
            ans, idx = [], []
            for t, ((p1, p2), (a1, a2)) in enumerate(zip(tr["players"], tr["answers"])):
                for pl, a in ((p1, a1), (p2, a2)):
                    if pl not in com:
                        ans.append(a == word); idx.append(t + 1)
            T = t50_stream(ans, idx, N); L = len(tr["outcome"])
            rows.append(dict(cm=cm, p=cm / N, T=float(T if T is not None else L), event=int(T is not None),
                             horizon=float(L if T is None else max(HORIZON, L)), order=f"{m[2]}{m[3]}", run=str(k)))
    return rows


# ------------------------------------------------------------------------------------------------------- AFT engine
def _tobit(Z, EV, Wt, iters=100):
    """Vectorised MLE of a censored normal location/scale for K problems. Z (K,n): log event times (EV) or log
    censoring times (~EV), already shifted by +gamma*log(p-p_c); Wt (K,n) 0/1 inclusion weights.
    Returns max log-lik (K,), a (K,), sigma (K,). Olsen parametrisation (b=a/s, th=1/s) -> globally concave."""
    ev = EV[None, :] * Wt; ce = (~EV)[None, :] * Wt
    ne = np.maximum(ev.sum(1), 1)
    mu = (Z * ev).sum(1) / ne
    sd = np.sqrt(((Z - mu[:, None]) ** 2 * ev).sum(1) / ne) + 0.05
    b, th = mu / sd, 1 / sd

    def ll(b, th):
        u = th[:, None] * Z - b[:, None]
        return (ev * (np.log(th)[:, None] - 0.5 * u ** 2 - HL2P)).sum(1) + (ce * log_ndtr(-u)).sum(1)

    cur = ll(b, th)
    for _ in range(iters):
        u = th[:, None] * Z - b[:, None]; v = -u
        lam = np.exp(-0.5 * v ** 2 - HL2P - log_ndtr(v)); h = -lam * (v + lam)
        gb = (ev * u + ce * lam).sum(1)
        gt = (ev * (1 / th[:, None] - u * Z) + ce * (-Z * lam)).sum(1)
        Hbb = (ev * -1.0 + ce * h).sum(1)
        Hbt = (ev * Z + ce * (-Z * h)).sum(1)
        Htt = (ev * (-1 / th[:, None] ** 2 - Z ** 2) + ce * (Z ** 2 * h)).sum(1)
        det = Hbb * Htt - Hbt ** 2
        db = -(Htt * gb - Hbt * gt) / det; dt = -(-Hbt * gb + Hbb * gt) / det
        step = np.ones_like(b); done = np.zeros(len(b), bool)
        for _ in range(40):
            nb, nt = b + step * db, th + step * dt
            ok = nt > 0
            new = ll(np.where(ok, nb, b), np.where(ok, nt, th)); new[~ok] = -np.inf
            acc = (~done) & ok & (new >= cur - 1e-12)
            b = np.where(acc, nb, b); th = np.where(acc, nt, th); cur = np.where(acc, new, cur); done |= acc
            if done.all():
                break
            step = np.where(done, step, step / 2)
        if np.nanmax(np.abs(np.r_[step * db, step * dt])) < 1e-9:
            break
    return cur, b / th, 1 / th


def _cens_normal_ll(y, ev):
    """Max log-lik of an iid censored normal (nuisance sub-threshold component); 0 if no events (sup)."""
    if len(y) == 0 or ev.sum() == 0:
        return 0.0
    L, _, _ = _tobit(y[None, :], ev, np.ones((1, len(y))))
    return float(L[0])


def pc_grid(pc_max, n=120):
    d = np.unique(np.r_[np.logspace(-5, np.log10(pc_max), n), pc_max])
    return np.unique(np.r_[0.0, pc_max - d[d <= pc_max]])  # dense near the bound, includes 0 (DIL)


def gamma_grid(step=0.01, gmax=4.0):
    return np.unique(np.round(np.r_[np.arange(step, gmax + 1e-9, step), 0.5, 1.0], 6))


def profile_surface(p, logT, ev, pcs, gs, mixture=False, chunk=40000):
    """L[i,j] = max_{a,sigma} loglik(p_c=pcs[i], gamma=gs[j]). mixture: runs with p <= p_c use a nuisance model."""
    P, G = len(pcs), len(gs)
    L = np.empty((P, G)); A = np.empty((P, G)); S = np.empty((P, G))
    rows_per = max(1, chunk // G)
    for i0 in range(0, P, rows_per):
        pc = pcs[i0:i0 + rows_per]
        above = p[None, :] > pc[:, None] + 1e-12                      # (r,n)
        x = np.log(np.where(above, p[None, :] - pc[:, None], 1.0))      # (r,n)
        Z = (logT[None, None, :] + gs[None, :, None] * x[:, None, :]).reshape(-1, len(p))
        Wt = np.repeat(above, G, axis=0).astype(float)
        l, a, s = _tobit(Z, ev, Wt)
        L[i0:i0 + rows_per] = l.reshape(-1, G); A[i0:i0 + rows_per] = a.reshape(-1, G); S[i0:i0 + rows_per] = s.reshape(-1, G)
        if mixture:
            for k, c in enumerate(pc):
                sub = p <= c + 1e-12
                L[i0 + k] += _cens_normal_ll(logT[sub], ev[sub])
        else:
            assert above.all(), "p_c grid must lie below every fitted p"
    return L, A, S


def nll_direct(theta, p, logT, ev):
    a, g, pc, ls = theta; s = np.exp(ls)
    if np.any(p <= pc):
        return 1e12
    z = (logT - (a - g * np.log(p - pc))) / s
    return -(np.sum(-0.5 * z[ev] ** 2 - HL2P - ls) + np.sum(log_ndtr(-z[~ev])))


def profile_ci(vals, prof, Lmax, lo_edge=None, hi_edge=None):
    thr = Lmax - CHI95 / 2
    ins = np.where(prof >= thr)[0]
    if len(ins) == 0:   # polished optimum lies outside the grid (e.g. p_c closer to the bound than the grid resolves)
        return dict(ci=[float("nan"), float("nan")], lo_at_edge=False, hi_at_edge=True, disjoint=False, optimum_off_grid=True)
    i0, i1 = ins[0], ins[-1]
    def interp(i, j):  # crossing between grid i (inside) and j (outside)
        return float(vals[i] + (thr - prof[i]) * (vals[j] - vals[i]) / (prof[j] - prof[i]))
    lo = interp(i0, i0 - 1) if i0 > 0 else float(vals[0])
    hi = interp(i1, i1 + 1) if i1 < len(vals) - 1 else float(vals[-1])
    gaps = bool(len(ins) != i1 - i0 + 1)
    return dict(ci=[lo, hi], lo_at_edge=bool(i0 == 0), hi_at_edge=bool(i1 == len(vals) - 1), disjoint=gaps)


def fit_aft(rows, cm_min, mixture=False, fine=True, polish=True):
    """Fit all four models; returns dict (point estimates, logliks, AIC, LR tests, profile CIs, profiles)."""
    R = [r for r in rows if r["cm"] >= cm_min]
    p = np.array([r["p"] for r in R]); logT = np.log([r["T"] for r in R]); ev = np.array([r["event"] for r in R], bool)
    pmin_above = min(r["p"] for r in R if r["cm"] >= (4 if mixture else cm_min))
    pc_max = pmin_above
    pcs = pc_grid(pc_max, 160 if fine else 60)[:-1]          # strictly below the bound
    gs = gamma_grid(0.01 if fine else 0.02, 4.0)
    L, A, S = profile_surface(p, logT, ev, pcs, gs, mixture=mixture)
    jh, j1 = int(np.where(np.isclose(gs, 0.5))[0][0]), int(np.where(np.isclose(gs, 1.0))[0][0])
    i, j = np.unravel_index(np.argmax(L), L.shape)
    crit = dict(ll=float(L[i, j]), a=float(A[i, j]), gamma=float(gs[j]), pc=float(pcs[i]), sigma=float(S[i, j]))
    if polish and not mixture:
        r0 = minimize(nll_direct, [crit["a"], crit["gamma"], crit["pc"], np.log(crit["sigma"])], args=(p, logT, ev),
                      method="L-BFGS-B", bounds=[(None, None), (0.001, 6), (0, pc_max - 1e-9), (-5, 3)])
        if -r0.fun > crit["ll"]:
            crit.update(ll=float(-r0.fun), a=float(r0.x[0]), gamma=float(r0.x[1]), pc=float(r0.x[2]), sigma=float(np.exp(r0.x[3])))
    def restricted(Lsub, kind):
        k = int(np.argmax(Lsub))
        if kind == "half": return dict(ll=float(L[k, jh]), pc=float(pcs[k]), a=float(A[k, jh]), sigma=float(S[k, jh]), gamma=0.5)
        if kind == "one": return dict(ll=float(L[k, j1]), pc=float(pcs[k]), a=float(A[k, j1]), sigma=float(S[k, j1]), gamma=1.0)
        return dict(ll=float(L[0, k]), pc=0.0, a=float(A[0, k]), sigma=float(S[0, k]), gamma=float(gs[k]))
    half, one, dil = restricted(L[:, jh], "half"), restricted(L[:, j1], "one"), restricted(L[0, :], "dil")
    if dil["gamma"] >= gs[-1] - 1e-9:
        dil["gamma_at_grid_edge"] = True
    nextra = 2 if mixture else 0
    models = dict(CRIT=(crit, 4), CRIT_half=(half, 3), CRIT_one=(one, 3), DIL=(dil, 3))
    out = dict(cm_min=cm_min, mixture=mixture, n_runs=len(R), n_censored=int((~ev).sum()), pc_upper_bound=float(pc_max),
               models={k: dict(v, k=kk + nextra, aic=float(-2 * v["ll"] + 2 * (kk + nextra))) for k, (v, kk) in models.items()})
    Lmax = max(crit["ll"], float(L.max()))
    prof_g, prof_pc = L.max(0), L.max(1)
    out["gamma_ci"] = profile_ci(gs, prof_g, Lmax)
    out["pc_ci"] = profile_ci(pcs, prof_pc, Lmax)
    out["pc_ci"]["hi_is_design_bound"] = out["pc_ci"]["hi_at_edge"]
    out["LR"] = {
        "gamma_free_vs_half": dict(stat=float(2 * (Lmax - half["ll"])), p=float(chi2.sf(2 * (Lmax - half["ll"]), 1))),
        "gamma_free_vs_one": dict(stat=float(2 * (Lmax - one["ll"])), p=float(chi2.sf(2 * (Lmax - one["ll"]), 1))),
        "threshold_vs_DIL": dict(stat=float(2 * (Lmax - dil["ll"])),
                                 p_boundary=float(0.5 * chi2.sf(max(2 * (Lmax - dil["ll"]), 0), 1))),
        "half_minus_one_loglik": float(half["ll"] - one["ll"]),
    }
    aics = {k: v["aic"] for k, v in out["models"].items()}
    out["dAIC"] = {k: float(v - min(aics.values())) for k, v in aics.items()}
    out["_profiles"] = dict(gamma=gs.tolist(), prof_gamma=(prof_g - Lmax).tolist(), pc=pcs.tolist(), prof_pc=(prof_pc - Lmax).tolist())
    return out


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------------------------- design simulation
def design(rows):
    """Claude archive design: list of (cm, horizon) per run."""
    return [(r["cm"], r["horizon"]) for r in rows]


def emu_T50(P, N, cm, horizon, seed):
    """One emulator run; T50 in interactions with the archive's 2N-answer window; censored at horizon."""
    half = N // 2
    sim = simulate(P, N, cm, max_rounds=int(np.ceil(horizon / half)) + 2, seed=seed)
    f, m = sim["f"], N - cm
    rng = np.random.default_rng(seed + 977)
    k = np.rint(f * m).astype(int)
    ans = (rng.random((len(f), m)).argsort(1) < k[:, None]).ravel()
    inter = (np.arange(len(f))[:, None] * half + np.floor(np.arange(m) * half / m)[None, :] + 1).ravel()
    T = t50_stream(ans, inter, N)
    return (float(T), 1) if (T is not None and T <= horizon) else (float(horizon), 0)


def sim_emulator_dataset(P, des, N, seed):
    out = []
    for i, (cm, hz) in enumerate(des):
        T, e = emu_T50(P, N, cm, hz, seed * 100_000 + i)
        out.append(dict(cm=cm, p=cm / N, T=T, event=e, horizon=hz))
    return out


def sim_synthetic_dataset(truth, des, N, seed):
    rng = np.random.default_rng(seed); out = []
    for cm, hz in des:
        p = cm / N
        if p > truth["pc"]:
            T = float(np.exp(truth["a"] - truth["gamma"] * np.log(p - truth["pc"]) + truth["sigma"] * rng.standard_normal()))
            T = max(T, 1.0)
        else:
            T = np.inf
        out.append(dict(cm=cm, p=p, T=min(T, hz), event=int(T <= hz), horizon=hz))
    return out


def _policy(name):
    if name == "best_response":
        return best_response_policy(5, 0.02)
    return load_policy(json.loads((ROOT / "results/policies/deepseek-v4-flash.json").read_text()))


def rec_worker(args):
    kind, seed, des, truth = args
    if kind in ("best_response", "deepseek"):
        rows = sim_emulator_dataset(_policy(kind), des, N_ARCH, seed)
    else:
        rows = sim_synthetic_dataset(truth, des, N_ARCH, seed)
    by = defaultdict(list)
    for r in rows: by[r["cm"]].append(r["event"])
    flips = {cm: float(np.mean(v)) for cm, v in sorted(by.items())}
    if sum(r["event"] for r in rows if r["cm"] >= 4) < 10:
        return dict(seed=seed, failed=True, flips=flips)
    f = fit_aft(rows, 4, fine=False, polish=False)
    return dict(seed=seed, failed=False, flips=flips, gamma=f["models"]["CRIT"]["gamma"], pc=f["models"]["CRIT"]["pc"],
                gamma_ci=f["gamma_ci"]["ci"], pc_ci=f["pc_ci"]["ci"], pc_hi_edge=f["pc_ci"]["hi_at_edge"],
                LR_half_p=f["LR"]["gamma_free_vs_half"]["p"], LR_one_p=f["LR"]["gamma_free_vs_one"]["p"],
                LR_dil_p=f["LR"]["threshold_vs_DIL"]["p_boundary"], dAIC=f["dAIC"],
                half_minus_one=f["LR"]["half_minus_one_loglik"], n_cens=f["n_censored"])


def summarize_rec(res, truth_gamma=None, truth_pc=None):
    ok = [r for r in res if not r["failed"]]
    g = np.array([r["gamma"] for r in ok]); pc = np.array([r["pc"] for r in ok])
    gci = np.array([r["gamma_ci"] for r in ok]); pci = np.array([r["pc_ci"] for r in ok])
    q = lambda v: [float(x) for x in np.percentile(v, [2.5, 25, 50, 75, 97.5])]
    best = [min(r["dAIC"], key=r["dAIC"].get) for r in ok]
    s = dict(n=len(res), n_ok=len(ok), gamma_q=q(g), gamma_mean=float(g.mean()), pc_q=q(pc),
             frac_pc_at_bound=float(np.mean([r["pc_hi_edge"] for r in ok])),
             frac_gamma_ci_covers_half=float(np.mean((gci[:, 0] <= .5) & (gci[:, 1] >= .5))),
             frac_gamma_ci_covers_one=float(np.mean((gci[:, 0] <= 1) & (gci[:, 1] >= 1))),
             frac_reject_half=float(np.mean([r["LR_half_p"] < .05 for r in ok])),
             frac_reject_one=float(np.mean([r["LR_one_p"] < .05 for r in ok])),
             frac_reject_DIL=float(np.mean([r["LR_dil_p"] < .05 for r in ok])),
             frac_half_beats_one_AIC=float(np.mean([r["half_minus_one"] > 0 for r in ok])),
             frac_half_beats_one_by2=float(np.mean([r["half_minus_one"] > 1 for r in ok])),   # dAIC > 2
             frac_one_beats_half_by2=float(np.mean([r["half_minus_one"] < -1 for r in ok])),
             best_model_freq={m: float(np.mean([b == m for b in best])) for m in ("CRIT", "CRIT_half", "CRIT_one", "DIL")},
             gamma_ci_median_width=float(np.median(gci[:, 1] - gci[:, 0])),
             mean_flip=dict((str(cm), float(np.mean([r["flips"][cm] for r in res]))) for cm in res[0]["flips"]))
    # "distinguish": reject the wrong exponent at 5% while not rejecting the right one
    s["frac_distinguish_half_from_one"] = float(np.mean([(r["LR_one_p"] < .05) and (r["LR_half_p"] >= .05) for r in ok]))
    s["frac_distinguish_one_from_half"] = float(np.mean([(r["LR_half_p"] < .05) and (r["LR_one_p"] >= .05) for r in ok]))
    if truth_gamma is not None:
        s.update(truth_gamma=truth_gamma, gamma_bias_median=float(np.median(g) - truth_gamma), gamma_bias_mean=float(g.mean() - truth_gamma),
                 gamma_coverage=float(np.mean((gci[:, 0] <= truth_gamma) & (gci[:, 1] >= truth_gamma))))
    if truth_pc is not None:
        s.update(truth_pc=truth_pc, pc_bias_median=float(np.median(pc) - truth_pc),
                 pc_coverage=float(np.mean((pci[:, 0] <= truth_pc + 1e-9) & (pci[:, 1] >= truth_pc - 1e-9))))
    s["_gamma_hat"] = g.tolist(); s["_pc_hat"] = pc.tolist()
    return s


# ------------------------------------------------------------------------- emulator "true" exponent (fine grid, big N)
def big_worker(args):
    kind, N, cm, R, horizon = args
    P = _policy(kind)
    res = [emu_T50(P, N, cm, horizon, 7_000_000 + 1000 * cm + s) for s in range(R)]
    return dict(N=N, cm=cm, p=cm / N, T=[t for t, _ in res], E=[e for _, e in res])


def emulator_truth(pool, kind="best_response", Ns=(24, 48, 96), R=150, horizon_rounds=3000):
    out = {}
    for N in Ns:
        cms = list(range(max(1, int(0.10 * N)), N // 2 + 1))
        jobs = [(kind, N, cm, R, horizon_rounds * N // 2) for cm in cms]
        cache = OUT / f"_emu_truth_{kind}_N{N}_R{R}.json"
        if cache.exists():
            lv = json.loads(cache.read_text())
        else:
            lv = pool.map(big_worker, jobs); cache.write_text(json.dumps(lv))
        flip = [(d["p"], float(np.mean(d["E"]))) for d in lv]
        pc_flip = None
        for (p0, f0), (p1, f1) in zip(flip, flip[1:]):
            if f0 < 0.5 <= f1:
                pc_flip = p0 + (0.5 - f0) * (p1 - p0) / (f1 - f0); break
        # AFT on all levels whose flip probability >= 0.5 (censored runs kept); p_c bounded by the lowest such level
        rows = [dict(cm=d["cm"], p=d["p"], T=t, event=e) for d in lv if np.mean(d["E"]) >= 0.5 for t, e in zip(d["T"], d["E"])]
        cm_min = min(r["cm"] for r in rows)
        f_all = fit_aft(rows, cm_min, fine=False, polish=True)
        # restricted to the archive-like range p in [p_c+, 0.5] but skipping the first level (less escape noise)
        f_skip = fit_aft([r for r in rows if r["cm"] > cm_min], cm_min + 1, fine=False, polish=True)
        med = [(d["p"], survival.km_median(d["T"], d["E"])) for d in lv]
        # local slopes d log median / d log(p - p_c_flip)
        loc = []
        if pc_flip is not None:
            pts = [(p, m) for p, m in med if p > pc_flip and np.isfinite(m)]
            for (p0, m0), (p1, m1) in zip(pts, pts[1:]):
                loc.append(dict(p_mid=(p0 + p1) / 2, dist_mid=(p0 + p1) / 2 - pc_flip,
                                gamma_local=float(-(np.log(m1) - np.log(m0)) / (np.log(p1 - pc_flip) - np.log(p0 - pc_flip)))))
        out[f"N{N}"] = dict(R=R, horizon_interactions=horizon_rounds * N // 2, pc_flip=pc_flip,
                            levels=[dict(cm=d["cm"], p=d["p"], flip=float(np.mean(d["E"])), km_median=m) for d, (_, m) in zip(lv, med)],
                            aft_all=dict(cm_min=cm_min, gamma=f_all["models"]["CRIT"]["gamma"], pc=f_all["models"]["CRIT"]["pc"],
                                         gamma_ci=f_all["gamma_ci"]["ci"], pc_ci=f_all["pc_ci"]["ci"], pc_at_bound=f_all["pc_ci"]["hi_at_edge"],
                                         dAIC=f_all["dAIC"]),
                            aft_skip_first=dict(cm_min=cm_min + 1, gamma=f_skip["models"]["CRIT"]["gamma"], pc=f_skip["models"]["CRIT"]["pc"],
                                                gamma_ci=f_skip["gamma_ci"]["ci"], pc_at_bound=f_skip["pc_ci"]["hi_at_edge"]),
                            local_gamma=loc)
        print(f"  [{kind} truth] N={N} pc_flip={pc_flip} AFT(all) gamma={f_all['models']['CRIT']['gamma']:.3f} "
              f"pc={f_all['models']['CRIT']['pc']:.4f} CI={np.round(f_all['gamma_ci']['ci'], 3).tolist()} | skip-first gamma="
              f"{f_skip['models']['CRIT']['gamma']:.3f}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------------------- main
def main(a):
    t0 = time.time()
    path = OUT / "study1_v2.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    rows = load_claude()
    by = defaultdict(list)
    for r in rows: by[r["cm"]].append(r)
    res["data"] = dict(source="Ashery et al. 2025 public archive, Claude, N=24, swap design, memory 5",
                       per_cm={str(cm): dict(p=v[0]["p"], n=len(v), n_tip=sum(r["event"] for r in v),
                                             censor_times=sorted(r["T"] for r in v if not r["event"])) for cm, v in sorted(by.items())})
    parts = a.parts.split(",")

    if "A" in parts:
        A = {}
        for name, cm_min, mix in (("primary_cm4", 4, False), ("mixture_cm2_all_runs", 2, True), ("drop_cm4_cm5", 5, False)):
            f = fit_aft(rows, cm_min, mixture=mix)
            if mix:   # does the data put cm=3 (p=0.125) above or below p_c?
                pcs_, pr_ = np.array(f["_profiles"]["pc"]), np.array(f["_profiles"]["prof_pc"])
                f["cm3_above_pc_best_rel_loglik"] = float(pr_[pcs_ < 3 / 24].max())
            A[name] = f
            m = f["models"]
            print(f"[A] {name}: n={f['n_runs']} cens={f['n_censored']} gamma={m['CRIT']['gamma']:.3f} CI={np.round(f['gamma_ci']['ci'], 3).tolist()} "
                  f"pc={m['CRIT']['pc']:.4f} CI={np.round(f['pc_ci']['ci'], 4).tolist()} (bound {f['pc_upper_bound']:.4f}, at edge={f['pc_ci']['hi_at_edge']}) "
                  f"dAIC={ {k: round(v, 2) for k, v in f['dAIC'].items()} } LR={ {k: (round(v['stat'], 2) if isinstance(v, dict) else round(v, 2)) for k, v in f['LR'].items()} }", flush=True)
        # comparison with the old least-squares, uncensored-only fit
        A["old_least_squares_uncensored_cm4"] = dict(gamma=0.53, gamma_boot_ci=[0.41, 0.69], pc=0.154, pc_boot_ci=[0.139, 0.162],
                                                    note="scripts/study1_time_to_tip.py (drops the censored cm=4 run)")
        res["A_aft"] = {k: (strip(v) if isinstance(v, dict) and "_profiles" in v else v) for k, v in A.items()}
        res["_A_profiles"] = {k: v["_profiles"] for k, v in A.items() if isinstance(v, dict) and "_profiles" in v}

    if "C" in parts:
        C = {}
        for cm, v in sorted(by.items()):
            T = [r["T"] for r in v]; E = [r["event"] for r in v]
            C[str(cm)] = dict(p=v[0]["p"], **survival.summarize(T, E, tau=HORIZON, n_boot=2000, seed=cm))
            s = C[str(cm)]
            print(f"[C] cm={cm:2d} n={s['n']} tips={s['n_event']} KM median={s['km_median']['est']} {s['km_median']['ci']}  "
                  f"RMST800={s['rmst']['est']:.1f} {np.round(s['rmst']['ci'], 1).tolist()}", flush=True)
        res["C_km_rmst"] = dict(tau=HORIZON, note="KM median = smallest t with S(t)<=0.5 (inf = not reached by horizon); "
                                "RMST = area under KM curve to tau=800 interactions (= mean min(T,800)); percentile bootstrap, 2000 reps",
                                per_cm=C)

    if "B" in parts:
        des = design(rows)
        prim = res["A_aft"]["primary_cm4"]["models"]
        truths = {f"synthetic_gamma{g}": dict(gamma=g, pc=prim[k]["pc"], a=prim[k]["a"], sigma=prim[k]["sigma"])
                  for g, k in ((0.5, "CRIT_half"), (1.0, "CRIT_one"))}
        # also a scenario where p_c sits well below the bound (truth not hugging cm_min/N)
        truths["synthetic_gamma0.5_pc0.125"] = dict(gamma=0.5, pc=0.125, a=prim["CRIT_half"]["a"], sigma=prim["CRIT_half"]["sigma"])
        truths["synthetic_gamma1.0_pc0.125"] = dict(gamma=1.0, pc=0.125, a=prim["CRIT_one"]["a"], sigma=prim["CRIT_one"]["sigma"])
        B = res.get("B_recovery", {})
        with Pool(a.workers) as pool:
            for kind, n in (("best_response", a.n_rec), ("deepseek", max(a.n_rec // 2, 50))):
                r = pool.map(rec_worker, [(kind, 1 + s, des, None) for s in range(n)])
                B[f"emulator_{kind}"] = summarize_rec(r)
                b = B[f"emulator_{kind}"]
                print(f"[B] emulator {kind}: ok {b['n_ok']}/{b['n']} gamma q={np.round(b['gamma_q'], 3).tolist()} pc q={np.round(b['pc_q'], 4).tolist()} "
                      f"rej1/2={b['frac_reject_half']:.2f} rej1={b['frac_reject_one']:.2f} pc@bound={b['frac_pc_at_bound']:.2f} flips={b['mean_flip']}", flush=True)
            for name, tr in truths.items():
                r = pool.map(rec_worker, [("synthetic", 10_000 + s, des, tr) for s in range(a.n_rec)])
                B[name] = summarize_rec(r, tr["gamma"], tr["pc"]); B[name]["truth"] = tr
                b = B[name]
                print(f"[B] {name}: gamma q={np.round(b['gamma_q'], 3).tolist()} bias={b['gamma_bias_median']:+.3f} cov={b['gamma_coverage']:.2f} "
                      f"pc bias={b['pc_bias_median']:+.4f} cov={b['pc_coverage']:.2f} rej1/2={b['frac_reject_half']:.2f} rej1={b['frac_reject_one']:.2f} "
                      f"distinguish(.5 vs 1)={b['frac_distinguish_half_from_one']:.2f}/{b['frac_distinguish_one_from_half']:.2f} pc@bound={b['frac_pc_at_bound']:.2f}", flush=True)
            res["B_recovery"] = {k: strip(v) for k, v in B.items()}
            res["_B_draws"] = {k: dict(gamma=v.get("_gamma_hat", []), pc=v.get("_pc_hat", [])) for k, v in B.items()}
    if "T" in parts:
        with Pool(a.workers) as pool:
            res["B_emulator_truth"] = emulator_truth(pool, "best_response", R=a.R_truth)

    res["meta"] = dict(script="scripts/study1_v2.py", runtime_s=round(time.time() - t0, 1), n_rec=a.n_rec)
    path.write_text(json.dumps(res, indent=1, default=float))
    if "fig" in parts:
        make_fig(res, rows)
    print(f"done in {time.time() - t0:.0f}s")


def make_fig(res, rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.titlelocation": "left",
                         "axes.titlesize": 9, "legend.frameon": False})
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
    prim = res["A_aft"]["primary_cm4"]; m = prim["models"]
    # (a) data + fits
    a0 = ax[0]
    jit = np.random.default_rng(0)
    for r in rows:
        x = r["p"] + jit.uniform(-0.004, 0.004)
        if r["event"]:
            a0.scatter(x, r["T"], s=7, color=GREY, alpha=0.55, lw=0)
        else:
            a0.scatter(x, r["T"], s=16, marker="^", facecolor="none", edgecolor=ORANGE, lw=0.7)
    C = res["C_km_rmst"]["per_cm"]
    for cm, s in C.items():
        med = s["km_median"]["est"]
        if np.isfinite(med):
            lo, hi = s["km_median"]["ci"]; hi = min(hi, 2000) if not np.isfinite(hi) else hi
            a0.errorbar(s["p"], med, yerr=[[med - lo], [hi - med]], fmt="s", ms=3, color="k", lw=0.7, capsize=0)
    pg = np.linspace(prim["pc_upper_bound"], 0.52, 300)
    for k, col, lab in (("CRIT", BLUE, "γ free"), ("CRIT_half", GREEN, "γ = ½"), ("CRIT_one", ORANGE, "γ = 1"), ("DIL", GREY, "no threshold")):
        mm = m[k]; pp = pg[pg > mm["pc"] + 1e-4]
        a0.plot(pp, np.exp(mm["a"] - mm["gamma"] * np.log(pp - mm["pc"])), color=col, lw=1.0,
                ls="--" if k == "DIL" else "-", label=f"{lab} (ΔAIC {prim['dAIC'][k]:.1f})")
    a0.axvline(prim["pc_upper_bound"], color=GREY, lw=0.5, ls=":")
    a0.set_yscale("log"); a0.set_ylim(40, 2500); a0.set_xlabel("committed fraction p"); a0.set_ylabel("T50 (interactions)")
    a0.set_title("a  Claude N=24: censored AFT fits (cm≥4)")
    a0.legend(fontsize=7, loc="upper right")
    a0.text(0.02, 0.03, "grey: tipped   orange ^: censored   black: KM median", transform=a0.transAxes, fontsize=6.5, color=GREY)
    # (b) profile likelihood for gamma
    a1 = ax[1]; pr = res["_A_profiles"]
    for k, col, lab in (("primary_cm4", BLUE, "cm ≥ 4 (primary; mixture overlaps)"), ("mixture_cm2_all_runs", GREEN, "all runs, sub-threshold mixture"),
                        ("drop_cm4_cm5", GREY, "cm ≥ 5 (drop cm=4)")):
        g = np.array(pr[k]["gamma"]); L = np.array(pr[k]["prof_gamma"])
        a1.plot(g, L, color=col, lw=1.0, label=lab)
    a1.axhline(-CHI95 / 2, color=GREY, lw=0.5, ls=":"); a1.text(2.45, -CHI95 / 2 + 0.15, "95% CI", fontsize=6.5, color=GREY)
    for v in (0.5, 1.0):
        a1.axvline(v, color="k", lw=0.4, ls="--")
    a1.set_xlim(0, 2.6); a1.set_ylim(-8, 0.4); a1.set_xlabel("γ"); a1.set_ylabel("profile log-likelihood (rel. max)")
    a1.set_title("b  profile likelihood of γ"); a1.legend(fontsize=7, loc="lower right")
    # (c) recovery distributions
    a2 = ax[2]; D = res["_B_draws"]; bins = np.arange(0.01, 2.0, 0.04)
    for k, col, lab in (("synthetic_gamma0.5", GREEN, "synthetic, true γ=½"), ("synthetic_gamma1.0", ORANGE, "synthetic, true γ=1"),
                        ("emulator_best_response", BLUE, "best-response emulator"), ("emulator_deepseek", GREY, "DeepSeek pooled emulator")):
        if k in D and D[k]["gamma"]:
            a2.hist(D[k]["gamma"], bins=bins, histtype="step", color=col, lw=1.0, density=True, label=lab)
    a2.axvline(m["CRIT"]["gamma"], color="k", lw=0.8)
    a2.text(m["CRIT"]["gamma"] + 0.03, a2.get_ylim()[1] * 0.55, f"Claude\nγ̂={m['CRIT']['gamma']:.2f}", fontsize=7)
    a2.set_xlim(0, 2.0)
    a2.set_xlabel("γ̂ (same AFT fit, Claude design)"); a2.set_ylabel("density")
    a2.set_title("c  parameter recovery at the archive design"); a2.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "fig3b_scaling_v2.png", dpi=200); plt.close(fig)
    print("figure ->", FIG / "fig3b_scaling_v2.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,C,B,T,fig"); ap.add_argument("--n-rec", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--R-truth", type=int, default=150)
    ap.add_argument("--skip-truth", action="store_true")
    main(ap.parse_args())
