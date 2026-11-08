"""Right-censored time-to-event helpers: Kaplan-Meier curve/median and restricted mean survival time (RMST),
with nonparametric bootstrap CIs (resampling runs). Pure numpy.

Conventions: T = event or censoring time, E = 1 if the event (tip) was observed, 0 if right-censored at T.
Ties: events at time t are processed before censorings at t (standard KM convention).
"""
from __future__ import annotations

import numpy as np


def km_curve(T, E):
    """Return (event_times, S(t) just after each event time). S is a right-continuous step function, S(0)=1."""
    T = np.asarray(T, float); E = np.asarray(E, int)
    ts = np.unique(T[E == 1])
    S, s = [], 1.0
    for t in ts:
        at_risk = np.sum(T >= t)
        d = np.sum((T == t) & (E == 1))
        s *= 1.0 - d / at_risk
        S.append(s)
    return ts, np.array(S)


def km_median(T, E):
    """Smallest t with S(t) <= 0.5; +inf if the KM curve never reaches 0.5 (median not estimable)."""
    ts, S = km_curve(T, E)
    hit = np.where(S <= 0.5)[0]
    return float(ts[hit[0]]) if len(hit) else float("inf")


def rmst(T, E, tau):
    """Restricted mean survival time: integral_0^tau S(t) dt (area under the KM step function)."""
    ts, S = km_curve(T, E)
    knots = np.r_[0.0, ts[ts < tau], tau]
    vals = np.r_[1.0, S[ts < tau]]
    return float(np.sum(np.diff(knots) * vals))


def km_survival_at(T, E, t):
    ts, S = km_curve(T, E)
    i = np.searchsorted(ts, t, side="right") - 1
    return 1.0 if i < 0 else float(S[i])


def bootstrap(T, E, stat, n_boot=2000, seed=0, level=0.95):
    """Percentile bootstrap CI for stat(T, E) resampling runs with replacement. Infinite replicates (e.g. KM median
    not reached) are kept, so an upper limit of inf means 'not estimable at this horizon'."""
    T = np.asarray(T, float); E = np.asarray(E, int); rng = np.random.default_rng(seed); n = len(T)
    reps = np.array([stat(T[i], E[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))], float)
    a = (1 - level) / 2
    lo, hi = np.quantile(reps, [a, 1 - a], method="lower"), np.quantile(reps, [a, 1 - a], method="higher")
    return dict(est=float(stat(T, E)), ci=[float(lo[0]), float(hi[1])], frac_inf=float(np.mean(~np.isfinite(reps))))


def summarize(T, E, tau, n_boot=2000, seed=0):
    """KM median + RMST(tau) with bootstrap CIs, plus counts."""
    T = np.asarray(T, float); E = np.asarray(E, int)
    return dict(n=int(len(T)), n_event=int(E.sum()), tau=float(tau),
                km_median=bootstrap(T, E, km_median, n_boot, seed),
                rmst=bootstrap(T, E, lambda t, e: rmst(t, e, tau), n_boot, seed + 1),
                surv_at_tau=km_survival_at(T, E, tau))
