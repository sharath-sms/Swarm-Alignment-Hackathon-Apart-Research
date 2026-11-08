"""Token-conditional policy extraction for ONE word pair (A = established convention, B = committed word).

Reuses cached pooled samples for that pair: in extract_policy/refine_policy the pair is WORD_PAIRS[k % 6], so new
samples use k ≡ j (mod 6) with fresh multiples -> identical tagging scheme, cache-compatible.
Steps: (1) per-pair policy from cache; (2) emulator visit counts at cms near threshold (N=24) under that policy;
(3) states covering `coverage` of visits get topped up to `target` samples; (4) write results/policies/<model>__<A><B>.json.
"""
from __future__ import annotations
import argparse, asyncio, json, random, re, sqlite3, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from swarm_ews.envs.naming_game import build_prompt, parse_value, _payoff  # noqa: E402
from swarm_ews.llm import LLMClient  # noqa: E402
from refine_policy import visits  # noqa: E402
WORD_PAIRS = [("Q", "M"), ("M", "Q"), ("X", "Y"), ("Y", "X"), ("F", "J"), ("J", "F")]
RECS = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]


def cached_counts(model, j):
    con = sqlite3.connect("data/cache/llm_cache.sqlite"); A, B = WORD_PAIRS[j]
    cnt = np.zeros((1024, 2)); kmax = np.full(1024, -1)
    for req, text in con.execute("select request, text from cache where request like ?", (f'%"tag": "policy/{model}/%',)):
        m = re.match(rf"policy/{re.escape(model)}/(\d{{5}})/k(\d+)", json.loads(req)["tag"])
        if not m or int(m.group(2)) % 6 != j: continue
        s, k = int(m.group(1), 4), int(m.group(2)); kmax[s] = max(kmax[s], k)
        v = parse_value(text, (A, B))
        if v is not None: cnt[s] += [v == B, 1]
    return cnt, kmax


async def main(a):
    j = WORD_PAIRS.index(tuple(a.pair)); A, B = WORD_PAIRS[j]; model = a.model
    cnt, kmax = cached_counts(model, j)
    pooled = json.loads(Path(f"results/policies/{model}.json").read_text())
    Pp = np.array([pooled["policy"]["".join(str((i // 4 ** (4 - d)) % 4) for d in range(5))]["P_B"] or 0.5 for i in range(1024)])
    P = np.where(cnt[:, 1] > 0, (cnt[:, 0] + 0.5) / (cnt[:, 1] + 1), Pp)
    V = visits(P, 24, [int(x) for x in a.cms.split(",")], a.runs, 300)
    order = np.argsort(-V); cum = np.cumsum(V[order]) / V.sum()
    chosen = order[: int(np.searchsorted(cum, a.coverage)) + 1]
    need = {int(s): max(0, a.target - int(cnt[s, 1])) for s in chosen}
    print(f"{model} {A}->{B}: {len(chosen)} states cover {a.coverage}; new samples {sum(need.values())}", flush=True)
    client = LLMClient(max_concurrency=32, spend_cap=a.cap)

    async def sample(s, k):
        digs = [(s // 4 ** (4 - d)) % 4 for d in range(5)]; mp = {"A": A, "B": B}
        hist = [(mp[RECS[i][0]], mp[RECS[i][1]], _payoff(RECS[i][0], RECS[i][1])) for i in digs]
        opts = [A, B]; random.Random(hash((tuple(digs), k))).shuffle(opts)
        instr, prompt = build_prompt(hist, opts)
        key = "".join(map(str, digs))
        r = await client.complete(model, prompt, instructions=instr, temperature=0.7, max_output_tokens=16,
                                  tag=f"policy/{model}/{key}/k{k}")
        v = parse_value(r.text, (A, B))
        if v is not None: cnt[s] += [v == B, 1]

    jobs = []
    for s, n in need.items():
        base = max(kmax[s], 47) + 1; base += (j - base) % 6     # next k ≡ j (mod 6), beyond pooled range
        jobs += [sample(s, base + 6 * m) for m in range(n)]
    await asyncio.gather(*jobs)
    out = dict(model=model, memory=5, K=None, temperature=0.7, pair=[A, B], records=["AA", "AB", "BA", "BB"],
               policy={"".join(str((i // 4 ** (4 - d)) % 4) for d in range(5)):
                       dict(P_B=float(cnt[i, 0] / cnt[i, 1]) if cnt[i, 1] else None, n=int(cnt[i, 1]), fail=0,
                            fallback_pooled=bool(cnt[i, 1] == 0)) for i in range(1024)},
               refined=dict(states=len(chosen), target=a.target, coverage=a.coverage, cms=a.cms))
    # states never sampled for this pair fall back to pooled P_B (flagged)
    for k, v in out["policy"].items():
        if v["P_B"] is None:
            v["P_B"] = pooled["policy"][k]["P_B"]; v["n"] = pooled["policy"][k]["n"]
    Path(f"results/policies/{model}__{A}{B}.json").write_text(json.dumps(out))
    print(client.summary(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--pair", nargs=2, required=True)
    ap.add_argument("--cms", default="2,3,4,5,6"); ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--coverage", type=float, default=0.99); ap.add_argument("--target", type=int, default=32)
    ap.add_argument("--cap", type=float, default=30.0)
    asyncio.run(main(ap.parse_args()))
