#!/usr/bin/env python3
"""OpenAI-compatible LLM API client with retry, concurrency control, and token tracking."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

# (input_$/M_token, output_$/M_token) — single source of truth for cost estimation
MODEL_RATES: dict[str, tuple[float, float]] = {
    "gpt-4o":           (2.50 / 1e6, 10.00 / 1e6),
    "gpt-4o-mini":      (0.15 / 1e6,  0.60 / 1e6),
    "claude-opus-4-6":  (5.00 / 1e6,  25.00 / 1e6),
    "claude-sonnet-4-6":(3.00 / 1e6,  15.00 / 1e6),
    "claude-haiku-4-5": (1.00 / 1e6,   5.00 / 1e6),
}
_DEFAULT_RATE = (2.50 / 1e6, 10.00 / 1e6)


@dataclass
class TokenUsage:
    """Tracks cumulative token usage and estimated cost."""
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    errors: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def add(self, input_t: int, output_t: int) -> None:
        async with self._lock:
            self.input_tokens += input_t
            self.output_tokens += output_t
            self.requests += 1

    async def add_error(self) -> None:
        async with self._lock:
            self.errors += 1

    def estimate_cost(self, model: str) -> float:
        """Rough cost estimate in USD."""
        in_rate, out_rate = MODEL_RATES.get(model, _DEFAULT_RATE)
        return self.input_tokens * in_rate + self.output_tokens * out_rate

    def summary(self, model: str) -> str:
        cost = self.estimate_cost(model)
        return (
            f"Requests: {self.requests} | Errors: {self.errors} | "
            f"Tokens: {self.input_tokens:,} in + {self.output_tokens:,} out | "
            f"Est. cost: ${cost:.2f}"
        )


_RETRYABLE = (APIError, APITimeoutError, RateLimitError)
_MAX_RETRIES = 3
_BASE_DELAY = 2.0


class LLMClient:
    """Async OpenAI-compatible client with concurrency control and retry."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        concurrency: int = 10,
        timeout: float = 120.0,
        json_mode: bool = True,
    ) -> None:
        self.model = model
        self.concurrency = concurrency
        self.json_mode = json_mode
        self.usage = TokenUsage()
        self._semaphore = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            timeout=timeout,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request with retry and concurrency control."""
        async with self._semaphore:
            last_err: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    kwargs: dict[str, Any] = dict(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if self.json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    resp = await self._client.chat.completions.create(**kwargs)
                    usage = resp.usage
                    if usage:
                        await self.usage.add(
                            usage.prompt_tokens, usage.completion_tokens,
                        )
                    return resp.choices[0].message.content or ""
                except _RETRYABLE as e:
                    last_err = e
                    await self.usage.add_error()
                    delay = _BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
            raise RuntimeError(
                f"LLM API failed after {_MAX_RETRIES} retries: {last_err}"
            )

    async def close(self) -> None:
        await self._client.close()
