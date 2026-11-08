"""Provider-agnostic async LLM client with caching, rate limiting, retries, and a hard spend cap.

Design goals for the experiments:
- Every agent call is independent: no server-side memory (agentic_memory=False, store=False).
- Every call is cached on disk (SQLite) keyed by (model, instructions, input, temperature,
  max_output_tokens, tag). The caller-supplied `tag` (e.g. "run=12/round=40/agent=3") makes
  repeated identical prompts at temperature > 0 distinct samples, while reruns hit the cache.
- Spend is tracked in a persistent ledger shared across processes; calls abort once the cap is hit.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Credits per 1M tokens (input, output). Source: docs.adaptionlabs.ai/inference-stacks/models (2026-11-06).
ADAPTION_PRICES: dict[str, tuple[float, float]] = {
    "gemma-4-31b-it": (0.09, 0.34),
    "gemma-4-26b-a4b-it": (0.0765, 0.255),
    "qwen3.8-27b": (0.039, 1.25),
    "deepseek-v4-flash": (0.04444, 0.13332),
    # "glm-53-flash": (0.15, 0.50),  # 404 "Model not found" for our key (2026-11-06)
    "qwen3.5-397b": (0.55, 3.50),
    "glm-5.2": (1.40, 4.40),
}

DEFAULT_DB = Path(os.environ.get("LLM_CACHE_DB", "data/cache/llm_cache.sqlite"))


class SpendCapExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost: float
    cached: bool
    model: str
    reasoning_tokens: int = 0


class _Store:
    """SQLite cache + spend ledger. Single-process async use; WAL allows concurrent readers."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, request TEXT, "
            "text TEXT, input_tokens INT, output_tokens INT, cost REAL, created REAL)"
        )
        self.conn.execute("CREATE TABLE IF NOT EXISTS ledger (ts REAL, model TEXT, cost REAL)")

    def get(self, key: str):
        return self.conn.execute(
            "SELECT text, input_tokens, output_tokens, cost, model FROM cache WHERE key=?", (key,)
        ).fetchone()

    def put(self, key, model, request, text, itok, otok, cost):
        self.conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?,?)",
            (key, model, json.dumps(request), text, itok, otok, cost, time.time()),
        )
        self.conn.execute("INSERT INTO ledger VALUES (?,?,?)", (time.time(), model, cost))

    def total_spend(self) -> float:
        return self.conn.execute("SELECT COALESCE(SUM(cost),0) FROM ledger").fetchone()[0]


class LLMClient:
    def __init__(
        self,
        provider: str = "adaption",
        max_concurrency: int = 16,
        spend_cap: float | None = None,
        db_path: Path = DEFAULT_DB,
        max_retries: int = 8,
        timeout: float = 20.0,
        rpm: int | None = None,
    ):
        self.provider = provider
        self.sem = asyncio.Semaphore(max_concurrency)
        self.spend_cap = float(spend_cap if spend_cap is not None else os.environ.get("SPEND_CAP", 50))
        self.store = _Store(Path(db_path))
        self.max_retries = max_retries
        self.timeout = timeout
        # Adaption enforces an account-wide limit of ~100 requests/minute (measured 2026-11-06; 429 'retry after
        # 60 seconds'). Client-side sliding-window limiter; run ONE experiment process at a time.
        self.rpm = int(rpm if rpm is not None else os.environ.get("RPM_LIMIT", 95))
        self._stamps = collections.deque()
        self._rl_lock = asyncio.Lock()
        self.n_retries = 0
        self._client = None
        self.session_spend = 0.0
        self.n_calls = 0
        self.n_cached = 0
        self.n_reasoning_leaks = 0

    def _get_client(self):
        if self._client is None:
            if self.provider == "adaption":
                from adaption import AsyncAdaption

                if not os.environ.get("ADAPTION_API_KEY"):
                    raise RuntimeError("ADAPTION_API_KEY not set (put it in .env at repo root).")
                self._client = AsyncAdaption(timeout=self.timeout, max_retries=0)
            else:
                raise ValueError(f"unknown provider {self.provider}")
        return self._client

    async def _acquire(self):
        async with self._rl_lock:
            while True:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] > 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self.rpm:
                    self._stamps.append(now)
                    return
                await asyncio.sleep(60.0 - (now - self._stamps[0]) + 0.05)

    @staticmethod
    def cost_of(model: str, itok: int, otok: int) -> float:
        pin, pout = ADAPTION_PRICES.get(model, (0.0, 0.0))
        return itok / 1e6 * pin + otok / 1e6 * pout

    @staticmethod
    def make_key(request: dict) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    async def complete(
        self,
        model: str,
        prompt: str,
        instructions: str | None = None,
        temperature: float = 0.8,
        max_output_tokens: int = 200,
        tag: str = "",
        reasoning_effort: str | None = "none",
    ) -> LLMResult:
        """reasoning_effort="none" disables thinking (verified 2026-11-06 for qwen3.8-27b, deepseek-v4-flash,
        qwen3.5-397b, glm-5.2: reasoning_tokens=0). Other provider knobs (enable_thinking, chat_template_kwargs,
        thinking.type) are ignored by the gateway. None = send nothing (model default, may think a lot)."""
        request = dict(
            provider=self.provider, model=model, instructions=instructions, input=prompt,
            temperature=temperature, max_output_tokens=max_output_tokens, tag=tag,
        )
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort
        key = self.make_key(request)
        hit = self.store.get(key)
        self.n_calls += 1
        if hit:
            self.n_cached += 1
            text, itok, otok, cost, m = hit
            return LLMResult(text, itok, otok, cost, True, m)

        if self.store.total_spend() >= self.spend_cap:
            raise SpendCapExceeded(f"spend cap {self.spend_cap} reached")

        client = self._get_client()
        async with self.sem:
            for attempt in range(self.max_retries):
                await self._acquire()
                try:
                    kwargs = dict(
                        model=model, input=prompt, temperature=temperature,
                        max_output_tokens=max_output_tokens, store=False, agentic_memory=False,
                    )
                    if instructions:
                        kwargs["instructions"] = instructions
                    if reasoning_effort is not None:
                        kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
                    resp = await client.responses.create(**kwargs)
                    break
                except Exception as e:  # retry on rate limits / transient errors
                    name = type(e).__name__
                    retriable = name in {"RateLimitError", "InternalServerError", "APIConnectionError",
                                         "APITimeoutError"}
                    if not retriable or attempt == self.max_retries - 1:
                        raise
                    self.n_retries += 1
                    if name == "RateLimitError":
                        await asyncio.sleep(61 + random.random() * 3)   # server window is 60 s
                    else:
                        await asyncio.sleep(min(30, 0.5 * 2 ** attempt + random.random()))

        usage = resp.usage
        itok = usage.input_tokens if usage else 0
        otok = usage.output_tokens if usage else 0
        details = (usage.output_tokens_details or {}) if usage else {}
        rtok = int(details.get("reasoning_tokens") or 0)
        if reasoning_effort == "none" and rtok > 0:
            self.n_reasoning_leaks += 1
        cost = self.cost_of(model, itok, otok)
        text = resp.output_text or ""
        self.store.put(key, model, request, text, itok, otok, cost)
        self.session_spend += cost
        return LLMResult(text, itok, otok, cost, False, model, rtok)

    def summary(self) -> str:
        return (f"calls={self.n_calls} cached={self.n_cached} session_spend={self.session_spend:.4f} "
                f"ledger_total={self.store.total_spend():.4f} cap={self.spend_cap} reasoning_leaks={self.n_reasoning_leaks} retries={self.n_retries}")


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from model text (tolerates code fences / prose)."""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:]
            if p.strip().startswith("{"):
                t = p
                break
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return None
