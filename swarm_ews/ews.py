"""Early-warning-signal (EWS) computation and evaluation.

Conventions follow the standard EWS literature (Dakos et al. 2012; ewstools):
- detrend with a Gaussian kernel whose sigma = bandwidth * len(series)
- rolling window = window_frac * len(series), indicators aligned to the window's right edge
- trend strength = Kendall tau of indicator vs time
Evaluation:
- run-level AUROC of tau separating tipping vs null runs (with bootstrap CI)
- surrogate significance (AR(1) surrogates fitted to residuals)
- online monitor: expanding-window tau, threshold calibrated on null runs at a target false-alarm rate
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau, skew
from sklearn.metrics import roc_auc_score

INDICATORS = ("variance", "ac1", "skew", "cv")
PRIMARY = ("variance", "ac1")


def detrend(x: np.ndarray, bandwidth: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, float)
    sigma = max(bandwidth * len(x), 1.0)
    trend = gaussian_filter1d(x, sigma=sigma, mode="nearest")
    return x - trend, trend


def _ac1(w: np.ndarray) -> float:
    a, b = w[:-1], w[1:]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rolling_indicators(x: np.ndarray, window_frac: float = 0.5, bandwidth: float = 0.2) -> dict[str, np.ndarray]:
    """Return dict indicator -> array (len(x)), NaN until the first full window."""
    x = np.asarray(x, float)
    n = len(x)
    resid, trend = detrend(x, bandwidth)
    w = max(int(round(window_frac * n)), 5)
    out = {k: np.full(n, np.nan) for k in INDICATORS}
    for t in range(w, n + 1):
        r = resid[t - w : t]
        out["variance"][t - 1] = r.var()
        out["ac1"][t - 1] = _ac1(r)
        out["skew"][t - 1] = skew(r) if r.std() > 0 else np.nan
        m = np.abs(trend[t - w : t]).mean()
        out["cv"][t - 1] = r.std() / m if m > 0 else np.nan
    return out


def trend_tau(ind: np.ndarray) -> float:
    m = ~np.isnan(ind)
    if m.sum() < 5:
        return np.nan
    tau, _ = kendalltau(np.arange(m.sum()), ind[m])
    return float(tau)


def ews_taus(x: np.ndarray, window_frac: float = 0.5, bandwidth: float = 0.2) -> dict[str, float]:
    ind = rolling_indicators(x, window_frac, bandwidth)
    return {k: trend_tau(v) for k, v in ind.items()}


# ---------- evaluation ----------

def auroc_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """AUROC with percentile bootstrap CI (resampling runs within each class)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    auc = roc_auc_score(y, np.r_[pos, neg])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        p = rng.choice(pos, len(pos))
        q = rng.choice(neg, len(neg))
        boots.append(roc_auc_score(y, np.r_[p, q]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def ar1_surrogates(x: np.ndarray, n: int = 200, bandwidth: float = 0.2, seed: int = 0) -> np.ndarray:
    """AR(1) surrogates matching the residuals' lag-1 autocorrelation and variance (stationary null)."""
    resid, _ = detrend(x, bandwidth)
    phi = np.clip(_ac1(resid), -0.99, 0.99)
    if np.isnan(phi):
        phi = 0.0
    sd = resid.std() * np.sqrt(1 - phi**2)
    rng = np.random.default_rng(seed)
    out = np.empty((n, len(x)))
    for i in range(n):
        e = rng.normal(0, sd, len(x))
        s = np.empty(len(x))
        s[0] = rng.normal(0, resid.std())
        for t in range(1, len(x)):
            s[t] = phi * s[t - 1] + e[t]
        out[i] = s
    return out


def surrogate_pvalues(x: np.ndarray, n: int = 200, window_frac: float = 0.5, bandwidth: float = 0.2,
                      seed: int = 0) -> dict[str, float]:
    obs = ews_taus(x, window_frac, bandwidth)
    sur = [ews_taus(s, window_frac, bandwidth) for s in ar1_surrogates(x, n, bandwidth, seed)]
    return {k: float((1 + sum(s[k] >= obs[k] for s in sur if not np.isnan(s[k]))) / (1 + len(sur)))
            for k in obs if not np.isnan(obs[k])}


def online_tau(x: np.ndarray, indicator: str, min_len: int = 40, step: int = 1,
               window_frac: float = 0.5, bandwidth: float = 0.2) -> np.ndarray:
    """tau(t) computed using only data up to t (causal). NaN before min_len."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for t in range(min_len, len(x) + 1, step):
        out[t - 1] = trend_tau(rolling_indicators(x[:t], window_frac, bandwidth)[indicator])
    return out


def calibrate_threshold(null_tau_traj: list[np.ndarray], far: float = 0.05) -> float:
    """Threshold such that a fraction `far` of null runs ever cross it (run-level false-alarm rate)."""
    maxima = [np.nanmax(t) for t in null_tau_traj if np.any(~np.isnan(t))]
    return float(np.quantile(maxima, 1 - far))


def first_alarm(tau_traj: np.ndarray, thr: float) -> int | None:
    idx = np.where(tau_traj > thr)[0]
    return int(idx[0]) if len(idx) else None
