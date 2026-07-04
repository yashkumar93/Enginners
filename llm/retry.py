"""Retry handler with cross-provider failover.

Implements the retry sequence:
    1. Current key on the preferred provider
    2. Next key(s) on the same provider (round-robin)
    3. Switch to fallback provider(s) in priority order
    4. Retry keys on each fallback provider

If every attempt fails, ``AllProvidersExhaustedError`` is raised.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from llm.config import MAX_RETRIES

if TYPE_CHECKING:
    from llm.cooldown import CooldownManager
    from llm.logger import RouterLogger
    from llm.metrics import RouterMetrics
    from llm.provider_pool import ProviderPool
    from llm.providers.base_provider import BaseProvider


# ------------------------------------------------------------------
# Custom exception
# ------------------------------------------------------------------


class AllProvidersExhaustedError(RuntimeError):
    """Raised when every provider and key has been tried and failed."""

    def __init__(self, agent_name: str, attempts: int) -> None:
        self.agent_name = agent_name
        self.attempts = attempts
        super().__init__(
            f"All providers exhausted for agent '{agent_name}' "
            f"after {attempts} attempt(s). Check API keys, rate limits, "
            "and provider availability."
        )


# ------------------------------------------------------------------
# Pattern for retryable errors
# ------------------------------------------------------------------

_RETRYABLE_PATTERNS: re.Pattern[str] = re.compile(
    r"429|rate.?limit|RateLimitError|Too Many Requests"
    r"|timeout|timed?\s*out|TimeoutError"
    r"|connect|connection|ConnectionError"
    r"|50[0-9]|InternalServerError|ServiceUnavailable|BadGateway",
    re.IGNORECASE,
)


def _is_retryable(error: BaseException) -> bool:
    """Return ``True`` if the error is transient and worth retrying."""
    error_text = f"{type(error).__name__}: {error}"
    return bool(_RETRYABLE_PATTERNS.search(error_text))


# ------------------------------------------------------------------
# Retry handler
# ------------------------------------------------------------------


class RetryHandler:
    """Orchestrates retry + failover logic across providers and keys.

    This class is stateless (no mutable instance state) — all shared
    state lives in the ``CooldownManager`` and ``RouterMetrics``
    singletons it references.
    """

    def __init__(
        self,
        provider_pool: ProviderPool,
        cooldown_manager: CooldownManager,
        logger: RouterLogger,
        metrics: RouterMetrics,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._pool = provider_pool
        self._cooldown = cooldown_manager
        self._logger = logger
        self._metrics = metrics
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute_with_retry(
        self,
        provider_name: str,
        model: str,
        agent_name: str,
        messages: Any,
        **kwargs: Any,
    ) -> str | Any:
        """Try to make a successful LLM call, with failover.

        Args:
            provider_name: Preferred provider for this agent.
            model:         Model identifier (without provider prefix).
            agent_name:    Human-readable agent name for logging.
            messages:      Messages payload forwarded to the LLM.
            **kwargs:      Extra arguments forwarded to ``LLM.call()``.

        Returns:
            The LLM's response string (or structured output).

        Raises:
            AllProvidersExhaustedError: When all retries are exhausted.
        """
        self._metrics.reset_retries(agent_name)

        # Build the ordered list of (provider, model) attempts
        attempts = self._build_attempt_sequence(provider_name, model)

        last_error: BaseException | None = None

        for attempt_num, (provider, attempt_model) in enumerate(attempts):
            if attempt_num >= self._max_retries:
                break

            # Try to get a non-cooldown key
            try:
                key_index, _api_key = provider.get_next_key()
            except RuntimeError:
                # All keys on this provider are on cooldown
                self._logger.debug(
                    f"All keys on cooldown for '{provider.name}', "
                    f"skipping to next provider"
                )
                continue

            start = time.monotonic()
            try:
                result = provider.call_llm(
                    model=attempt_model,
                    key_index=key_index,
                    messages=messages,
                    **kwargs,
                )
                latency = time.monotonic() - start

                # ✓ Success
                self._metrics.record_request(
                    provider=provider.name,
                    key_index=key_index,
                    success=True,
                    latency=latency,
                )
                self._logger.log_request(
                    agent_name=agent_name,
                    provider=provider.name,
                    key_index=key_index,
                    model=attempt_model,
                    latency=latency,
                    retry_count=attempt_num,
                    success=True,
                )
                return result

            except Exception as exc:
                latency = time.monotonic() - start
                last_error = exc
                failure_reason = f"{type(exc).__name__}: {exc}"

                # Record failure
                self._metrics.record_request(
                    provider=provider.name,
                    key_index=key_index,
                    success=False,
                    latency=latency,
                )
                self._metrics.record_retry(agent_name)
                self._logger.log_request(
                    agent_name=agent_name,
                    provider=provider.name,
                    key_index=key_index,
                    model=attempt_model,
                    latency=latency,
                    retry_count=attempt_num,
                    success=False,
                    failure_reason=failure_reason,
                )

                if _is_retryable(exc):
                    # Put the key on cooldown
                    self._cooldown.mark_cooldown(provider.name, key_index)
                    self._metrics.mark_cooldown(provider.name, key_index)
                    self._logger.log_cooldown_activated(
                        provider=provider.name,
                        key_index=key_index,
                        reason=failure_reason,
                    )
                else:
                    # Non-retryable → propagate immediately
                    raise

        # All attempts exhausted
        self._logger.log_all_exhausted(agent_name)
        raise AllProvidersExhaustedError(
            agent_name=agent_name,
            attempts=min(len(attempts), self._max_retries),
        ) from last_error

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_attempt_sequence(
        self,
        primary_provider_name: str,
        primary_model: str,
    ) -> list[tuple[BaseProvider, str]]:
        """Build an ordered list of ``(provider, model)`` pairs to try.

        Order:
            1. Primary provider keys (multiple rounds via round-robin)
            2. Fallback providers (each with their own key rotation)

        The list length may exceed ``max_retries``; the caller truncates.
        """
        attempts: list[tuple[BaseProvider, str]] = []
        primary = self._pool.get_provider(primary_provider_name)

        # Add primary provider attempts (one per key, giving each key a chance)
        primary_key_count = len(primary.keys)
        for _ in range(max(primary_key_count, 1)):
            attempts.append((primary, primary_model))

        # Add fallback provider attempts
        fallbacks = self._pool.get_fallback_providers(exclude=primary_provider_name)
        for fb_provider in fallbacks:
            self._logger.log_provider_switch(
                from_provider=primary_provider_name,
                to_provider=fb_provider.name,
                agent_name="(building attempt sequence)",
            )
            # Pick the first model from the fallback provider's supported list
            fb_model = fb_provider.models[0] if fb_provider.models else primary_model
            fb_key_count = len(fb_provider.keys)
            for _ in range(max(fb_key_count, 1)):
                attempts.append((fb_provider, fb_model))

        return attempts
