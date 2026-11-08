"""Probe how to disable/limit reasoning tokens on Adaption models (tiny spend, not cached)."""
import asyncio, sys, json
from dotenv import load_dotenv; load_dotenv()
from adaption import AsyncAdaption

PROMPT = ('Two players each pick a word, "Q" or "M". Matching earns points. Your partner picked "M" last time. '
          'Reply with JSON only: {"word": "<Q or M>"}')
VARIANTS = {
    "baseline": {},
    "reasoning_effort_none": {"reasoning": {"effort": "none"}},
    "reasoning_effort_minimal": {"reasoning": {"effort": "minimal"}},
    "enable_thinking_false": {"enable_thinking": False},
    "chat_template_kwargs": {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}},
    "thinking_disabled": {"thinking": {"type": "disabled"}},
}
async def main(models):
    c = AsyncAdaption()
    for m in models:
        for name, extra in VARIANTS.items():
            try:
                r = await c.responses.create(model=m, input=PROMPT, temperature=0.7, max_output_tokens=600,
                                             store=False, agentic_memory=False, extra_body=extra or None)
                types = [o.type for o in (r.output or [])]
                u = r.usage
                print(f"{m:18s} {name:26s} out={u.output_tokens:4d} details={u.output_tokens_details} "
                      f"items={types} text={ (r.output_text or '').strip()[:60]!r}")
            except Exception as e:
                print(f"{m:18s} {name:26s} ERROR {type(e).__name__}: {str(e)[:120]}")
asyncio.run(main(sys.argv[1:]))
