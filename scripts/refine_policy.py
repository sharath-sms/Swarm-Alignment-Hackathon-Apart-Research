"""Study 2a (refinement): adaptive extra sampling of the memory states that dominate dynamics near threshold.

1) Run the emulator (current policy) at committed counts around the predicted critical mass and count how often
   non-committed agents occupy each memory state. 2) Take the smallest set of states covering `coverage` of visits.
3) Draw K2 extra samples for each (tags k = K..K+K2-1, disjoint from the first pass). 4) Merge counts and save.
"""
from __future__ import annotations
import argparse, asyncio, json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.emulator import load_policy  # noqa: E402
from swarm_ews.envs.naming_game import build_prompt, parse_value, _payoff  # noqa: E402
from swarm_ews.llm import LLMClient  # noqa: E402
WORD_PAIRS = [("Q", "M"), ("M", "Q"), ("X", "Y"), ("Y", "X"), ("F", "J"), ("J", "F")]
RECS = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]

def visits(P, N, cms, runs, rounds, memory=5, seed=0):
    rng = np.random.default_rng(seed); V = np.zeros(4 ** memory); mod = 4 ** memory
    for cm in cms:
        for s in range(runs):
            com = np.zeros(N, bool); com[rng.choice(N, cm, replace=False)] = True
            st = np.zeros(N, np.int64)
            for t in range(rounds):
                np.add.at(V, st[~com], 1)
                ch = np.where(com, 1, (rng.random(N) < P[st]).astype(int))
                perm = rng.permutation(N); pt = np.empty(N, int); pt[perm[0::2]] = perm[1::2]; pt[perm[1::2]] = perm[0::2]
                st = (st * 4 + 2 * ch + ch[pt]) % mod
                if ch[~com].mean() >= 0.95: break
    return V

async def main(a):
    js = json.loads(Path(a.policy).read_text()); P = load_policy(js); m = js["memory"]; model = js["model"]
    V = np.zeros(4 ** m)
    for N, cms in ((24, [int(x) for x in a.cms24.split(",")]), (48, [int(x) for x in a.cms48.split(",")])):
        V += visits(P, N, cms, a.runs, a.rounds) / N
    order = np.argsort(-V); cum = np.cumsum(V[order]) / V.sum()
    chosen = order[: int(np.searchsorted(cum, a.coverage)) + 1]
    print(f"{model}: {len(chosen)} states cover {a.coverage:.3f} of visits; extra samples = {len(chosen) * a.K2}", flush=True)
    client = LLMClient(max_concurrency=32, spend_cap=a.cap)
    pol = js["policy"]; K0 = js["K"]
    def key_of(idx): return "".join(str((idx // 4 ** (m - 1 - k)) % 4) for k in range(m))
    extra = {key_of(i): [0, 0, 0] for i in chosen}
    async def sample(key, k):
        s = [int(c) for c in key]; A, B = WORD_PAIRS[k % 6]; mp = {"A": A, "B": B}
        hist = [(mp[RECS[i][0]], mp[RECS[i][1]], _payoff(RECS[i][0], RECS[i][1])) for i in s]
        opts = [A, B]; random.Random(hash((tuple(s), k))).shuffle(opts)
        instr, prompt = build_prompt(hist, opts)
        r = await client.complete(model, prompt, instructions=instr, temperature=js["temperature"], max_output_tokens=16,
                                  tag=f"policy/{model}/{key}/k{k}")
        v = parse_value(r.text, (A, B)); c = extra[key]
        if v is None: c[2] += 1
        else: c[0] += v == B; c[1] += 1
    await asyncio.gather(*(sample(key, k) for key in extra for k in range(K0, K0 + a.K2)))
    for key, (nb, n, fail) in extra.items():
        old = pol[key]; onb = round((old["P_B"] or 0) * old["n"]); tot = old["n"] + n
        pol[key] = dict(P_B=(onb + nb) / tot if tot else None, n=tot, fail=old["fail"] + fail)
    js["refined"] = dict(states=len(chosen), K2=a.K2, coverage=a.coverage, cms24=a.cms24, cms48=a.cms48)
    Path(a.policy).write_text(json.dumps(js))
    print(client.summary(), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True); ap.add_argument("--cms24", default="2,3,4,5"); ap.add_argument("--cms48", default="5,6,7,8")
    ap.add_argument("--runs", type=int, default=100); ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--coverage", type=float, default=0.995); ap.add_argument("--K2", type=int, default=40)
    ap.add_argument("--cap", type=float, default=30.0)
    asyncio.run(main(ap.parse_args()))
