"""Smoke test: one tiny call per model to verify the key, model availability, JSON compliance,
latency, and real token usage / cost. Spend is negligible (< 0.01 credits total).

Usage: .venv/bin/python scripts/check_models.py [model ...]
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_ews.llm import ADAPTION_PRICES, LLMClient, parse_json  # noqa: E402

PROMPT = (
    "You manage a fishing boat. The lake currently holds 62 tons of fish. Last round the group caught 15 tons. "
    "How many tons (integer 0-10) do you catch this round? "
    'Reply with JSON only: {"harvest": <int>, "reason": "<max 15 words>"}'
)


async def main(models):
    client = LLMClient(spend_cap=1.0, db_path=Path("data/cache/smoke.sqlite"))
    for m in models:
        t0 = time.time()
        try:
            r = await client.complete(m, PROMPT, temperature=0.8, max_output_tokens=120, tag=f"smoke/{time.time()}")
            js = parse_json(r.text)
            print(f"[OK] {m:22s} {time.time()-t0:5.1f}s in={r.input_tokens:4d} out={r.output_tokens:4d} "
                  f"cost={r.cost:.6f} json={'yes' if js else 'NO'} -> {r.text.strip()[:90]!r}")
        except Exception as e:
            print(f"[FAIL] {m:22s} {type(e).__name__}: {str(e)[:150]}")
    print(client.summary())


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(ADAPTION_PRICES)))
