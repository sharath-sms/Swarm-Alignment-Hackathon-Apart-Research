"""Synthetic positive / negative controls for validating the EWS pipeline (no LLM).

Positive control: stochastic logistic growth with constant harvest H and slowly decreasing growth rate r(t).
The fold (saddle-node) bifurcation occurs at r* = 4H/K; approaching it from above produces critical slowing down.
Negative control: stationary AR(1) noise (no approaching transition).
"""
from __future__ import annotations

import numpy as np


def harvested_logistic(T: int = 400, K: float = 100.0, H: float = 10.0, r_start_mult: float = 2.0,
                       r_end_mult: float = 0.5, ramp: bool = True, sigma: float = 1.0, seed: int = 0,
                       tip_frac: float = 0.1, dt: float = 1.0):
    """Returns (x, r, tip_index or None). Ramped runs tip after r falls below r*; null runs hold r fixed."""
    rng = np.random.default_rng(seed)
    r_star = 4 * H / K
    r = np.linspace(r_start_mult * r_star, r_end_mult * r_star, T) if ramp else np.full(T, r_start_mult * r_star)
    # start at the upper equilibrium of the initial r
    r0 = r[0]
    x = np.empty(T)
    x[0] = K / 2 * (1 + np.sqrt(max(1 - 4 * H / (r0 * K), 0)))
    for t in range(1, T):
        drift = r[t - 1] * x[t - 1] * (1 - x[t - 1] / K) - H
        x[t] = max(x[t - 1] + dt * drift + sigma * np.sqrt(dt) * rng.normal(), 0.0)
    tip = None
    below = x < tip_frac * K
    for t in range(T - 5):
        if below[t : t + 5].all():
            tip = t
            break
    return x, r, tip


def ar1(T: int = 400, phi: float = 0.5, sigma: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = 0.0
    for t in range(1, T):
        x[t] = phi * x[t - 1] + sigma * rng.normal()
    return x


def proportional_logistic_rampK(T: int = 400, r: float = 0.5, h: float = 0.25, K_start: float = 100.0,
                                K_end: float = 30.0, sigma: float = 1.0, seed: int = 0):
    """MEAN-DRIFT NULL (methodology review F1): logistic growth with *proportional* harvest h*x and slowly
    decreasing carrying capacity K(t). Equilibrium x* = K(1 - h/r) falls with K, but the linearised recovery
    rate (r - h) is constant -> no critical slowing down, no bifurcation. EWS should NOT fire here; a
    level/trend monitor will (the mean declines just like in tipping runs)."""
    rng = np.random.default_rng(seed)
    K = np.linspace(K_start, K_end, T)
    x = np.empty(T)
    x[0] = K[0] * (1 - h / r)
    for t in range(1, T):
        drift = r * x[t - 1] * (1 - x[t - 1] / K[t - 1]) - h * x[t - 1]
        x[t] = max(x[t - 1] + drift + sigma * rng.normal(), 0.0)
    return x, K
