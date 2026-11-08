"""Study 2a: extract each model's stochastic policy P(choose B | memory state) by sampling.

Memory state = ordered tuple of `memory` records, each (my word, partner word) in {A,B}^2 (payoff is implied).
With memory=5 there are 4^5 = 1024 states. For each state we draw K samples at the experiment temperature,
cycling through counterbalanced word pairs (which letter is A vs B) and shuffled option orders, using the exact
prompt builder of the live environment. Output: results/policies/<model>.json with P_B, n, n_fail per state.
(Adaption exposes no logprobs, so probabilities are estimated by sampling; Flint et al. 2026 used logprobs.)
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.envs.naming_game import build_prompt, parse_value, _payoff  # noqa: E402
from swarm_ews.llm import LLMClient  # noqa: E402

WORD_PAIRS = [("Q", "M"), ("M", "Q"), ("X", "Y"), ("Y", "X"), ("F", "J"), ("J", "F")]
RECS = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]


def states(memory):
    return list(itertools.product(range(4), repeat=memory))


async def main(a):
    out = Path("results/policies"); out.mkdir(parents=True, exist_ok=True)
    client = LLMClient(max_concurrency=a.concurrency, spend_cap=a.cap)
    for model in a.models.split(","):
        S = states(a.memory)
        counts = {s: [0, 0, 0] for s in S}  # nB, n, fail

        async def sample(s, k):
            A, B = WORD_PAIRS[k % len(WORD_PAIRS)]
            m = {"A": A, "B": B}
            hist = [(m[RECS[i][0]], m[RECS[i][1]], _payoff(RECS[i][0], RECS[i][1])) for i in s]
            opts = [A, B]; random.Random(hash((s, k))).shuffle(opts)
            instr, prompt = build_prompt(hist, opts)
            r = await client.complete(model, prompt, instructions=instr, temperature=a.temperature,
                                      max_output_tokens=16, tag=f"policy/{model}/{''.join(map(str, s))}/k{k}")
            v = parse_value(r.text, (A, B))
            c = counts[s]
            if v is None:
                c[2] += 1
            else:
                c[0] += v == B; c[1] += 1

        await asyncio.gather(*(sample(s, k) for s in S for k in range(a.K)))
        res = {"model": model, "memory": a.memory, "K": a.K, "temperature": a.temperature,
               "records": ["AA", "AB", "BA", "BB"],
               "policy": {"".join(map(str, s)): dict(P_B=(c[0] / c[1]) if c[1] else None, n=c[1], fail=c[2])
                          for s, c in counts.items()}}
        (out / f"{model}.json").write_text(json.dumps(res))
        fails = sum(c[2] for c in counts.values())
        print(f"{model}: states={len(S)} samples={len(S)*a.K} fails={fails} | {client.summary()}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--memory", type=int, default=5)
    p.add_argument("--K", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--cap", type=float, default=30.0)
    asyncio.run(main(p.parse_args()))
