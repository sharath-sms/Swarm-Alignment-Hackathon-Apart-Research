"""Target-conditional policy extraction for a NEW model and a fixed word pair (A = established, B = pushed).

Stage 1: K1 samples for every memory state (1024), random option order, fixed words.
Stage 2: emulate N=24 populations at cm in `cms` with the stage-1 MLE policy, take the states covering `coverage` of
non-committed visits, top up each to `target` samples. MLE (unsmoothed) is used throughout (see estimator study).
Output: results/policies/<model>__<A><B>.json  (extract_policy-compatible).
Tags: policy3/<model>/<A><B>/<state>/k<k>  (disjoint from earlier extractions).
"""
from __future__ import annotations
import argparse, asyncio, json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarm_ews.envs.naming_game import build_prompt, parse_value, _payoff  # noqa: E402
from swarm_ews.llm import LLMClient  # noqa: E402
from refine_policy import visits  # noqa: E402
RECS = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]


async def main(a):
    A, B = a.pair; model = a.model; client = LLMClient(max_concurrency=32, spend_cap=a.cap)
    cnt = np.zeros((1024, 2)); fails = np.zeros(1024)

    async def sample(s, k):
        digs = [(s // 4 ** (4 - d)) % 4 for d in range(5)]; mp = {"A": A, "B": B}
        hist = [(mp[RECS[i][0]], mp[RECS[i][1]], _payoff(RECS[i][0], RECS[i][1])) for i in digs]
        opts = [A, B]; random.Random(hash((tuple(digs), k, A, B))).shuffle(opts)
        instr, prompt = build_prompt(hist, opts)
        r = await client.complete(model, prompt, instructions=instr, temperature=0.7, max_output_tokens=24,
                                  tag=f"policy3/{model}/{A}{B}/{''.join(map(str, digs))}/k{k}")
        v = parse_value(r.text, (A, B))
        if v is None: fails[s] += 1
        else: cnt[s] += [v == B, 1]

    await asyncio.gather(*(sample(s, k) for s in range(1024) for k in range(a.K1)))
    print(f"stage 1 done: {int(cnt[:,1].sum())} samples, fails {int(fails.sum())} | {client.summary()}", flush=True)
    P = np.where(cnt[:, 1] > 0, cnt[:, 0] / np.maximum(cnt[:, 1], 1), 0.5)
    V = visits(P, 24, [int(x) for x in a.cms.split(",")], a.runs, 300)
    order = np.argsort(-V); cum = np.cumsum(V[order]) / V.sum()
    chosen = order[: int(np.searchsorted(cum, a.coverage)) + 1]
    jobs = [sample(int(s), a.K1 + m) for s in chosen for m in range(max(0, a.target - int(cnt[s, 1])))]
    print(f"stage 2: {len(chosen)} states cover {a.coverage}; {len(jobs)} top-up samples", flush=True)
    await asyncio.gather(*jobs)
    key = lambda i: "".join(str((i // 4 ** (4 - d)) % 4) for d in range(5))
    out = dict(model=model, memory=5, K=a.K1, temperature=0.7, pair=[A, B], records=["AA", "AB", "BA", "BB"],
               policy={key(i): dict(P_B=float(cnt[i, 0] / cnt[i, 1]) if cnt[i, 1] else 0.5, n=int(cnt[i, 1]), fail=int(fails[i]))
                       for i in range(1024)},
               refined=dict(states=len(chosen), target=a.target, coverage=a.coverage, cms=a.cms))
    Path(f"results/policies/{model}__{A}{B}.json").write_text(json.dumps(out))
    print("saved |", client.summary(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--pair", nargs=2, required=True)
    ap.add_argument("--K1", type=int, default=8); ap.add_argument("--target", type=int, default=24)
    ap.add_argument("--coverage", type=float, default=0.95); ap.add_argument("--cms", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--runs", type=int, default=100); ap.add_argument("--cap", type=float, default=40.0)
    asyncio.run(main(ap.parse_args()))
