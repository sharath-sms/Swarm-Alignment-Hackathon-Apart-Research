"""LLM naming game with committed minority, memory poisoning, and pulse perturbations.

Protocol (adapted from Ashery, Aiello & Baronchelli 2025, MIT-licensed code; changes documented):
- N agents (all the same LLM unless `model_of` overrides). Each round, a uniformly random perfect matching pairs all
  agents; every pair plays one coordination game simultaneously (parallel dynamics as in Flint et al. 2026).
  One round = N/2 interactions.
- Each agent sees only its own last `memory` records: (its word, partner's word, payoff). No cumulative score,
  no round counter beyond the window, fixed prompt length once the window is full -> no hidden state drift.
- Options are presented in a freshly shuffled order on every call; payoff +100 match / -50 mismatch.
- Initial state: every agent's memory pre-filled with `memory` successful rounds of convention A.
- Committed agents (count cm) always play B and are never queried.
- Memory poisoning (hidden driver): each round, each non-committed agent's new record is forged with probability
  q(t): the partner's word is recorded as B and the payoff recomputed consistently. Agents cannot see this.
- Pulse perturbation: at given rounds, forge ALL non-committed records for that round (q=1 for one round).
- Reasoning disabled (effort "none"); output must be JSON {"value": "<word>"}; invalid -> one resample, then a
  random word (logged as parse failure).
Observable per round: f_t = fraction of non-committed agents that played B.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable

from ..llm import LLMClient

RULES = (
    "You are playing a repeated partnership game. In each round you are paired with a partner and you both "
    "simultaneously choose one value from: {options}.\n"
    "If you both choose the SAME value, you each get +100 points. If you choose DIFFERENT values, you each get "
    "-50 points. Your goal is to maximize your own points."
)
QUERY = (
    "Your history in recent rounds (oldest first):\n{history}\n\n"
    "Which value do you choose this round? Reply with JSON only: {{\"value\": \"<one of {options}>\"}}"
)


@dataclass
class NGConfig:
    model: str
    N: int = 48
    cm: int = 0
    memory: int = 5
    words: tuple[str, str] = ("Q", "M")       # (convention A, committed word B)
    temperature: float = 0.7
    max_rounds: int = 150
    q_schedule: Callable[[int], float] | None = None   # round -> poisoning prob
    pulse_rounds: tuple[int, ...] = ()
    stop_on_tip: bool = True                 # stop when f >= 0.9 for 5 consecutive rounds
    min_rounds: int = 0                      # never stop before this round
    seed: int = 0
    tag: str = "ng"
    max_output_tokens: int = 16


@dataclass
class AgentState:
    committed: bool
    hist: list = field(default_factory=list)  # list of (mine, partner, payoff)


def _payoff(a, b):
    return 100 if a == b else -50


def build_prompt(hist, options):
    h = "\n".join(f"- You chose {m}, partner chose {p}, payoff {'+' if s > 0 else ''}{s}" for m, p, s in hist)
    opt = "[" + ", ".join(options) + "]"
    return RULES.format(options=opt), QUERY.format(history=h, options=opt)


_VAL = re.compile(r'"value"\s*:\s*"?([A-Za-z]+)"?')


def parse_value(text: str, words) -> str | None:
    m = _VAL.search(text or "")
    if m and m.group(1) in words:
        return m.group(1)
    toks = [w for w in words if re.search(rf"\b{re.escape(w)}\b", text or "")]
    return toks[0] if len(toks) == 1 else None


async def run_game(cfg: NGConfig, client: LLMClient) -> dict:
    rng = random.Random(cfg.seed)
    A, B = cfg.words
    agents = [AgentState(committed=False, hist=[(A, A, 100)] * cfg.memory) for _ in range(cfg.N)]
    for i in rng.sample(range(cfg.N), cfg.cm):
        agents[i].committed = True
    noncom = [i for i, a in enumerate(agents) if not a.committed]
    rounds, n_parse_fail, n_calls, cost, itok, otok, leaks0 = [], 0, 0, 0.0, 0, 0, client.n_reasoning_leaks
    run_above = 0

    async def choose(i, t):
        nonlocal n_parse_fail, n_calls, cost, itok, otok
        a = agents[i]
        if a.committed:
            return B
        for attempt in range(2):
            opts = [A, B]
            rng_local = random.Random(hash((cfg.seed, t, i, attempt)))
            rng_local.shuffle(opts)
            instr, prompt = build_prompt(a.hist[-cfg.memory:], opts)
            r = await client.complete(cfg.model, prompt, instructions=instr, temperature=cfg.temperature,
                                      max_output_tokens=cfg.max_output_tokens,
                                      tag=f"{cfg.tag}/s{cfg.seed}/t{t}/a{i}/try{attempt}")
            n_calls += 1; cost += r.cost if not r.cached else 0.0; itok += r.input_tokens; otok += r.output_tokens
            v = parse_value(r.text, (A, B))
            if v is not None:
                return v
        n_parse_fail += 1
        return rng.choice([A, B])

    for t in range(cfg.max_rounds):
        order = list(range(cfg.N)); rng.shuffle(order)
        pairs = [(order[k], order[k + 1]) for k in range(0, cfg.N - 1, 2)]
        choices = await asyncio.gather(*(choose(i, t) for i in range(cfg.N)))
        q = cfg.q_schedule(t) if cfg.q_schedule else 0.0
        pulse = t in cfg.pulse_rounds
        n_forged = 0
        for i, j in pairs:
            for me, other in ((i, j), (j, i)):
                if agents[me].committed:
                    continue
                pw = choices[other]
                if pulse or (q > 0 and rng.random() < q):
                    pw = B; n_forged += 1
                agents[me].hist.append((choices[me], pw, _payoff(choices[me], pw)))
                agents[me].hist = agents[me].hist[-cfg.memory:]
        f = sum(choices[i] == B for i in noncom) / max(len(noncom), 1)
        succ = sum(choices[i] == choices[j] for i, j in pairs) / len(pairs)
        rounds.append(dict(t=t, f=f, success=succ, q=q, pulse=pulse, n_forged=n_forged))
        run_above = run_above + 1 if f >= 0.9 else 0
        if cfg.stop_on_tip and run_above >= 5 and t + 1 >= cfg.min_rounds:
            break

    return dict(config=dict(model=cfg.model, N=cfg.N, cm=cfg.cm, p=cfg.cm / cfg.N, memory=cfg.memory,
                            words=list(cfg.words), temperature=cfg.temperature, max_rounds=cfg.max_rounds,
                            pulse_rounds=list(cfg.pulse_rounds), seed=cfg.seed, tag=cfg.tag),
                rounds=rounds, n_rounds=len(rounds), n_calls=n_calls, n_parse_fail=n_parse_fail,
                parse_fail_rate=n_parse_fail / max(n_calls, 1), new_cost=cost, input_tokens=itok,
                output_tokens=otok, reasoning_leaks=client.n_reasoning_leaks - leaks0)
