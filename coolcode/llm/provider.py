"""Multi-provider LLM interface with automatic failover and cost-based routing."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from coolcode.config import CoolCodeConfig, LLMConfig

logger = logging.getLogger(__name__)

# Errors worth retrying (transient: rate limits, network hiccups, 5xx)
_RETRY_ERROR_SUBSTRINGS = (
    "rate limit",
    "rate_limit",
    "429",
    "503",
    "502",
    "504",
    "timeout",
    "connection",
    "temporarily unavailable",
    "overloaded",
    "try again",
)

# Errors that should NOT be retried (auth, invalid request, context too long)
_PERMANENT_ERROR_SUBSTRINGS = (
    "401",
    "403",
    "invalid api key",
    "authentication",
    "invalid_request",
    "context length",
    "max_tokens",
    "context_length_exceeded",
)


def _is_retryable(exc: Exception) -> bool:
    """Heuristic: decide if an LLM exception is transient and worth retrying."""
    msg = str(exc).lower()
    if any(s in msg for s in _PERMANENT_ERROR_SUBSTRINGS):
        return False
    if any(s in msg for s in _RETRY_ERROR_SUBSTRINGS):
        return True
    # Unknown errors — try once more, but not beyond
    return True

# Cost per 1M tokens (input, output)
COST_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "MiniMax-M2.7-highspeed": (0.5, 2.0),
    "MiniMax-M2.7": (1.0, 4.0),
    "MiniMax-M2.5": (0.15, 1.1),
    "MiniMax-M2.5-highspeed": (0.3, 2.2),
}


@dataclass
class ProviderStats:
    """Track provider performance for intelligent routing."""

    total_calls: int = 0
    total_failures: int = 0
    total_latency_ms: float = 0.0
    last_failure_time: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_calls, 1)

    @property
    def failure_rate(self) -> float:
        return self.total_failures / max(self.total_calls, 1)


def _init_claude(config: LLMConfig) -> BaseChatModel:
    """Initialize a Claude model."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=config.model,
        api_key=config.api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )


def _init_minimax(config: LLMConfig) -> BaseChatModel:
    """Initialize a MiniMax model."""
    from langchain_community.chat_models import MiniMaxChat

    return MiniMaxChat(
        model=config.model,
        minimax_api_key=config.api_key,
        minimax_group_id=config.group_id,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


_INIT_MAP = {
    "claude": _init_claude,
    "minimax": _init_minimax,
}


def create_llm(config: LLMConfig) -> BaseChatModel:
    """Create a LangChain chat model from provider config."""
    init_fn = _INIT_MAP.get(config.provider)
    if init_fn is None:
        raise ValueError(f"Unknown provider: {config.provider}")
    return init_fn(config)


class LLMProvider:
    """Multi-provider LLM manager with failover, cost routing, and stats tracking.

    Supports two routing strategies:
    - "cost": prefer the cheapest provider (MiniMax first, Claude fallback)
    - "quality": prefer the highest-quality provider (Claude first, MiniMax fallback)
    - "fast": prefer the provider with lowest average latency
    """

    def __init__(self, app_config: CoolCodeConfig, strategy: str = "cost"):
        self.config = app_config
        self.strategy = strategy
        self._models: dict[str, BaseChatModel] = {}
        self._stats: dict[str, ProviderStats] = {}

        for provider_cfg in app_config.providers:
            key = f"{provider_cfg.provider}:{provider_cfg.model}"
            self._models[key] = create_llm(provider_cfg)
            self._stats[key] = ProviderStats()

    @property
    def available_providers(self) -> list[str]:
        return list(self._models.keys())

    def _rank_providers(self) -> list[str]:
        """Rank providers based on the current strategy."""
        keys = list(self._models.keys())

        if self.strategy == "cost":
            def cost_key(k: str) -> float:
                model = k.split(":", 1)[1]
                inp, out = COST_TABLE.get(model, (10.0, 50.0))
                return inp + out
            keys.sort(key=cost_key)

        elif self.strategy == "quality":
            quality_order = {"claude": 0, "minimax": 1}
            keys.sort(key=lambda k: quality_order.get(k.split(":")[0], 99))

        elif self.strategy == "fast":
            keys.sort(key=lambda k: self._stats[k].avg_latency_ms)

        # Deprioritize providers that recently failed
        now = time.time()
        keys.sort(key=lambda k: 1 if (now - self._stats[k].last_failure_time) < 60 else 0)

        return keys

    def get_model(self, provider_key: str | None = None) -> BaseChatModel:
        """Get a specific model or the top-ranked one."""
        if provider_key and provider_key in self._models:
            return self._models[provider_key]
        ranked = self._rank_providers()
        if not ranked:
            raise RuntimeError("No LLM providers available.")
        return self._models[ranked[0]]

    def get_all_models(self) -> list[tuple[str, BaseChatModel]]:
        """Return all available models as (provider_key, model) pairs.

        Used by the swarm to spawn workers across ALL providers in parallel.
        """
        return [(key, model) for key, model in self._models.items()]

    async def invoke_with_failover(
        self,
        messages: list[BaseMessage],
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        **kwargs: Any,
    ) -> Any:
        """Invoke the best-ranked model, retrying transient errors with exponential
        backoff, then falling over to the next provider on persistent failure.

        Retry strategy per provider:
            attempt 1: immediate
            attempt 2: base_delay * 2^0 + jitter
            attempt 3: base_delay * 2^1 + jitter
            ...

        Permanent errors (auth, invalid request) skip retries and move on.
        """
        ranked = self._rank_providers()
        last_error: Exception | None = None

        for key in ranked:
            model = self._models[key]
            stats = self._stats[key]

            for attempt in range(max_retries):
                start = time.monotonic()
                try:
                    result = await model.ainvoke(messages, **kwargs)
                    elapsed = (time.monotonic() - start) * 1000
                    stats.total_calls += 1
                    stats.total_latency_ms += elapsed
                    if attempt > 0:
                        logger.info(
                            f"[{key}] success in {elapsed:.0f}ms (after {attempt} retries)"
                        )
                    else:
                        logger.info(f"[{key}] success in {elapsed:.0f}ms")
                    return result
                except Exception as e:
                    elapsed = (time.monotonic() - start) * 1000
                    stats.total_calls += 1
                    stats.total_failures += 1
                    stats.total_latency_ms += elapsed
                    stats.last_failure_time = time.time()
                    last_error = e

                    if not _is_retryable(e):
                        logger.warning(
                            f"[{key}] permanent error ({e}), moving to next provider"
                        )
                        break

                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        delay += random.uniform(0, delay * 0.25)  # +0-25% jitter
                        logger.warning(
                            f"[{key}] transient error ({e}), retry {attempt + 1}/{max_retries} "
                            f"in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.warning(
                            f"[{key}] exhausted {max_retries} retries, moving to next provider"
                        )

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Return provider performance stats."""
        return {
            key: {
                "calls": s.total_calls,
                "failures": s.total_failures,
                "failure_rate": f"{s.failure_rate:.1%}",
                "avg_latency_ms": f"{s.avg_latency_ms:.0f}",
            }
            for key, s in self._stats.items()
        }
