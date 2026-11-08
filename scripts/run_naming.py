"""Run a grid of naming-game experiments. Resumable: completed runs (JSON file present) are skipped.

Examples:
  python scripts/run_naming.py --exp pilot --models gemma-4-31b-it --N 48 --cms 2,4,6,8,12,16 --seeds 3 --max-rounds 120
  python scripts/run_naming.py --exp smoke --models deepseek-v4-flash --N 8 --cms 2 --seeds 1 --max-rounds 5
Word pairs and which word is committed are counterbalanced across seeds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.envs.naming_game import NGConfig, run_game  # noqa: E402
from swarm_ews.llm import LLMClient, SpendCapExceeded  # noqa: E402

WORD_PAIRS = [("Q", "M"), ("M", "Q"), ("X", "Y"), ("Y", "X"), ("F", "J"), ("J", "F")]


def ramp(q0, q1, t0, t1, hold_after=True):
    def f(t):
        if t <= t0:
            return q0
        if t >= t1:
            return q1 if hold_after else q0
        return q0 + (q1 - q0) * (t - t0) / (t1 - t0)
    return f


async def main(a):
    out = Path("results/runs") / a.exp
    out.mkdir(parents=True, exist_ok=True)
    client = LLMClient(max_concurrency=a.concurrency, spend_cap=a.cap)
    jobs = []
    for model in a.models.split(","):
        for cm in [int(x) for x in a.cms.split(",")]:
            for s in range(a.seed_offset, a.seed_offset + a.seeds):
                words = tuple(a.words.split(",")) if a.words else WORD_PAIRS[s % len(WORD_PAIRS)]
                q_sched, qdesc = None, None
                if a.q_ramp:
                    q0, q1, t0, t1 = [float(x) for x in a.q_ramp.split(",")]
                    q_sched, qdesc = ramp(q0, q1, int(t0), int(t1)), a.q_ramp
                pulses = tuple(int(x) for x in a.pulses.split(",")) if a.pulses else ()
                name = f"{model}__N{a.N}__cm{cm}__s{s}" + (f"__w{words[0]}{words[1]}" if a.words else "") + (f"__q{qdesc.replace(',', '_')}" if qdesc else "")
                f = out / f"{name}.json"
                if f.exists():
                    continue
                cfg = NGConfig(model=model, N=a.N, cm=cm, memory=a.memory, words=words, temperature=a.temperature,
                               max_rounds=a.max_rounds, q_schedule=q_sched, pulse_rounds=pulses,
                               stop_on_tip=not a.no_stop, min_rounds=a.min_rounds, seed=s,
                               tag=f"{a.exp}/{model}/N{a.N}/cm{cm}" + (f"/w{words[0]}{words[1]}" if a.words else "") + (f"/q{qdesc}" if qdesc else ""))
                jobs.append((cfg, f, qdesc))
    print(f"{len(jobs)} runs to do -> {out}", flush=True)
    sem = asyncio.Semaphore(a.parallel_runs)
    t0 = time.time()
    done = 0

    async def one(cfg, f, qdesc):
        nonlocal done
        async with sem:
            try:
                res = await run_game(cfg, client)
            except SpendCapExceeded as e:
                print("SPEND CAP:", e, flush=True); return
            except Exception as e:
                print(f"[ERR] {f.name}: {type(e).__name__}: {str(e)[:200]}", flush=True); return
            res["config"]["q_ramp"] = qdesc
            f.write_text(json.dumps(res))
            done += 1
            last = res["rounds"][-1]["f"] if res["rounds"] else None
            print(f"[{done}/{len(jobs)}] {f.stem} rounds={res['n_rounds']} f_end={last:.2f} "
                  f"parse_fail={res['parse_fail_rate']:.3f} cost={res['new_cost']:.4f} "
                  f"ledger={client.store.total_spend():.3f} ({time.time()-t0:.0f}s)", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    print(client.summary(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--N", type=int, default=48)
    p.add_argument("--cms", default="0")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--memory", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-rounds", type=int, default=150)
    p.add_argument("--min-rounds", type=int, default=0)
    p.add_argument("--no-stop", action="store_true")
    p.add_argument("--words", default=None, help="fixed A,B (convention, committed word)")
    p.add_argument("--q-ramp", default=None, help="q0,q1,t0,t1")
    p.add_argument("--pulses", default=None)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--parallel-runs", type=int, default=8)
    p.add_argument("--cap", type=float, default=50.0)
    asyncio.run(main(p.parse_args()))
