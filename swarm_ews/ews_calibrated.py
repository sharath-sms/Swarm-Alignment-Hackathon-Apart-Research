"""Null-calibrated EWS indicators for binomial-proportion series (fix for external critique F1).

Problem: the binomial-normalised variance  var(resid) / [p(1-p)/n_obs]  (p = Gaussian-kernel trend) has a Kendall-tau
trend that depends on the *level path* of p even for pure independent binomial noise (e.g. biased positive at low p,
strongly positive when p rises). Raw AC1 on near-0/1 proportion series has similar level-path artefacts (constant
windows, discreteness).

Fix: per-series parametric surrogates. Fit the series' own smooth trend (same Gaussian detrend as ews.detrend,
bandwidth 0.2), draw M independent Binomial(n_obs, trend_t)/n_obs series, recompute the indicator's Kendall tau on each,
and report the standardised score z = (tau_obs - mean_sur) / sd_sur (and the surrogate percentile). Under the null of
independent binomial sampling around a smooth trend, z is ~ centred at 0 whatever the level path.

Indicator definitions are identical to the originals:
  - var : scripts/emulator_study.binom_norm_var_tau  (window int(0.5 n), trend window-mean clipped to [1e-3, 1-1e-3])
  - ac1 : ews.rolling_indicators(...)["ac1"]         (window round(0.5 n), NaN on constant windows)
All rolling computations and Kendall taus are vectorised over surrogates.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d


def _detrend_rows(X: np.ndarray, bandwidth: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    sigma = max(bandwidth * X.shape[1], 1.0)
    tr = gaussian_filter1d(X, sigma=sigma, axis=1, mode="nearest")
    return X - tr, tr


def kendall_tau_time(V: np.ndarray) -> np.ndarray:
    """Kendall tau-b of each row of V against time, ignoring NaNs (same as scipy.kendalltau(arange, row[~nan]))."""
    V = np.atleast_2d(V)
    d = V[:, None, :] - V[:, :, None]                 # d[m, i, j] = v_j - v_i
    K = V.shape[1]
    upper = np.triu(np.ones((K, K), bool), 1)
    valid = ~np.isnan(d) & upper
    s = np.where(valid, np.sign(np.nan_to_num(d)), 0.0).sum(axis=(1, 2))
    n0 = valid.sum(axis=(1, 2)).astype(float)
    ties = (valid & (d == 0)).sum(axis=(1, 2)).astype(float)
    nvalid = (~np.isnan(V)).sum(axis=1)
    den = np.sqrt(n0 * (n0 - ties))
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where((den > 0) & (nvalid >= 5), s / den, np.nan)
    return tau


def binom_var_series(X: np.ndarray, n_obs: int, bandwidth: float = 0.2) -> np.ndarray:
    """Rolling binomial-normalised variance for each row (matches emulator_study.binom_norm_var_tau)."""
    X = np.atleast_2d(np.asarray(X, float))
    resid, tr = _detrend_rows(X, bandwidth)
    w = max(int(0.5 * X.shape[1]), 5)
    rv = sliding_window_view(resid, w, axis=1).var(axis=2)
    tm = np.clip(sliding_window_view(tr, w, axis=1).mean(axis=2), 1e-3, 1 - 1e-3)
    return rv / (tm * (1 - tm) / n_obs)


def ac1_series(X: np.ndarray, bandwidth: float = 0.2) -> np.ndarray:
    """Rolling lag-1 autocorrelation for each row (matches ews.rolling_indicators()['ac1'], NaN on constant windows)."""
    X = np.atleast_2d(np.asarray(X, float))
    resid, _ = _detrend_rows(X, bandwidth)
    w = max(int(round(0.5 * X.shape[1])), 5)
    W = sliding_window_view(resid, w, axis=1)     # (M, K, w)
    a, b = W[..., :-1], W[..., 1:]
    a = a - a.mean(axis=2, keepdims=True); b = b - b.mean(axis=2, keepdims=True)
    sa, sb = np.sqrt((a ** 2).sum(axis=2)), np.sqrt((b ** 2).sum(axis=2))
    num = (a * b).sum(axis=2)
    eps = 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((sa > eps) & (sb > eps), num / (sa * sb), np.nan)
    return r


def raw_taus(f: np.ndarray, n_obs: int, bandwidth: float = 0.2) -> dict[str, float]:
    f = np.asarray(f, float)[None, :]
    return dict(var=float(kendall_tau_time(binom_var_series(f, n_obs, bandwidth))[0]),
                ac1=float(kendall_tau_time(ac1_series(f, bandwidth))[0]))


def binomial_surrogates(f: np.ndarray, n_obs: int, M: int, rng: np.random.Generator,
                        bandwidth: float = 0.2) -> np.ndarray:
    _, tr = _detrend_rows(np.asarray(f, float)[None, :], bandwidth)
    p = np.clip(tr[0], 0.0, 1.0)
    return rng.binomial(n_obs, p, size=(M, len(p))) / n_obs


def calibrated_scores(f: np.ndarray, n_obs: int, M: int = 100, seed: int = 0, bandwidth: float = 0.2,
                      sur_bandwidth: float | None = None) -> dict:
    """Observed taus, surrogate-standardised z and surrogate percentile for variance and AC1.

    `bandwidth` is the indicator's detrending bandwidth (fixed 0.2 as in the originals). `sur_bandwidth` (default =
    bandwidth) is the bandwidth of the trend the surrogates are drawn around; a smaller value (e.g. 0.05) reproduces
    the trend-misfit residual that sharp, accelerating mean changes leave after 0.2-detrending (sensitivity option)."""
    f = np.asarray(f, float)
    rng = np.random.default_rng(seed)
    S = binomial_surrogates(f, n_obs, M, rng, bandwidth if sur_bandwidth is None else sur_bandwidth)
    X = np.vstack([f[None, :], S])
    out = {}
    for name, ser in (("var", binom_var_series(X, n_obs, bandwidth)), ("ac1", ac1_series(X, bandwidth))):
        taus = kendall_tau_time(ser)
        obs, sur = taus[0], taus[1:]
        sur = sur[~np.isnan(sur)]
        out[f"{name}_tau"] = float(obs)
        if np.isnan(obs) or len(sur) < 10 or sur.std() < 1e-9:
            out[f"{name}_z"] = np.nan; out[f"{name}_pct"] = np.nan
        else:
            out[f"{name}_z"] = float((obs - sur.mean()) / sur.std(ddof=1))
            out[f"{name}_pct"] = float((np.sum(sur < obs) + 0.5 * np.sum(sur == obs)) / len(sur))
    return out
