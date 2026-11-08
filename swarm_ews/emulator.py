"""Markov emulator of the LLM naming game using an extracted (or rule-based) policy table.

Same protocol as envs/naming_game.py: random perfect matching each round, memory window of fixed length,
committed agents always play B, optional memory poisoning q(t) and pulses. Vectorised over agents; pure numpy.
State encoding: record code = 2*myB + partnerB  (0=AA, 1=AB, 2=BA, 3=BB); state index = base-4 number of the
window, oldest record most significant (matches extract_policy.py's itertools.product order).
"""
from __future__ import annotations

import numpy as np


def load_policy(js: dict) -> np.ndarray:
    m = js["memory"]
    P = np.full(4 ** m, np.nan)
    for k, v in js["policy"].items():
        idx = int(k, 4) if len(k) == m else int("".join(k), 4)
        P[idx] = v["P_B"] if v["P_B"] is not None else np.nan
    return np.nan_to_num(P, nan=0.5)


def best_response_policy(memory: int = 5, eps: float = 0.02, beta: float = None) -> np.ndarray:
    """Classical reference agent: best-respond to partner frequencies in memory (ties -> own last word), with
    trembling-hand noise eps. Payoffs +100/-50 -> choose B iff partner B-share > 1/3? (EV_B = 100s - 50(1-s);
    EV_A = 100(1-s) - 50s -> B iff s > 1/2). Ties broken by own last choice."""
    P = np.empty(4 ** memory)
    for idx in range(4 ** memory):
        digs = [(idx // 4 ** (memory - 1 - k)) % 4 for k in range(memory)]
        sB = np.mean([d & 1 for d in digs])
        own_last_B = digs[-1] >> 1
        pb = 1.0 if sB > 0.5 else 0.0 if sB < 0.5 else float(own_last_B)
        P[idx] = (1 - eps) * pb + eps * 0.5
    return P


def simulate(P: np.ndarray, N: int, cm: int, memory: int = 5, max_rounds: int = 300, q_schedule=None,
             pulse_rounds=(), stop_on_tip: bool = True, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    committed = np.zeros(N, bool); committed[rng.choice(N, cm, replace=False)] = True
    state = np.zeros(N, np.int64)            # all-AA history -> index 0
    mod = 4 ** memory
    f_hist, run_above = [], 0
    nc = ~committed
    for t in range(max_rounds):
        pB = P[state]
        choice = np.where(committed, 1, (rng.random(N) < pB).astype(int))
        perm = rng.permutation(N); partner = np.empty(N, int)
        partner[perm[0::2]] = perm[1::2]; partner[perm[1::2]] = perm[0::2]
        pw = choice[partner].copy()
        q = q_schedule(t) if q_schedule else 0.0
        if t in pulse_rounds:
            pw[:] = 1
        elif q > 0:
            pw[rng.random(N) < q] = 1
        rec = 2 * choice + pw
        state = (state * 4 + rec) % mod
        f = choice[nc].mean()
        f_hist.append(f)
        run_above = run_above + 1 if f >= 0.9 else 0
        if stop_on_tip and run_above >= 5:
            break
    return dict(f=np.array(f_hist), n_rounds=len(f_hist))


def t50(f: np.ndarray, win: int = 2) -> int | None:
    """First round where the rolling mean (window `win` rounds) of f reaches 0.5."""
    if len(f) < win:
        return None
    r = np.convolve(f, np.ones(win) / win, mode="valid")
    h = np.where(r >= 0.5)[0]
    return int(h[0] + win - 1) if len(h) else None
