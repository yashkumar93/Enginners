"""Thread-safe metrics tracking for the LLM Router.

Records per-provider and per-key statistics so operators can monitor
rate-limit pressure, latency trends, and failure rates at a glance.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


class RouterMetrics:
    """Accumulates request metrics in a thread-safe manner.

    All public methods acquire ``_lock`` before mutating or reading
    internal counters so the class is safe to use from concurrent
    CrewAI agent threads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # provider → count
        self._requests_per_provider: dict[str, int] = defaultdict(int)

        # (provider, key_index) → count
        self._requests_per_key: dict[tuple[str, int], int] = defaultdict(int)

        self._success_count: int = 0
        self._failure_count: int = 0

        # Running sums for computing averages
        self._total_latency: float = 0.0
        self._latency_count: int = 0

        # Current retry state (reset per-request cycle)
        self._current_retries: dict[str, int] = defaultdict(int)

        # (provider, key_index) → bool — mirrors CooldownManager
        self._active_cooldowns: set[tuple[str, int]] = set()

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_request(
        self,
        *,
        provider: str,
        key_index: int,
        success: bool,
        latency: float,
    ) -> None:
        """Record the outcome of a single LLM call attempt."""
        with self._lock:
            self._requests_per_provider[provider] += 1
            self._requests_per_key[(provider, key_index)] += 1
            if success:
                self._success_count += 1
            else:
                self._failure_count += 1
            self._total_latency += latency
            self._latency_count += 1

    def record_retry(self, agent_name: str) -> None:
        """Increment the retry counter for *agent_name*."""
        with self._lock:
            self._current_retries[agent_name] += 1

    def reset_retries(self, agent_name: str) -> None:
        """Reset the retry counter after a request cycle completes."""
        with self._lock:
            self._current_retries[agent_name] = 0

    def mark_cooldown(self, provider: str, key_index: int) -> None:
        """Mark a key as being on cooldown."""
        with self._lock:
            self._active_cooldowns.add((provider, key_index))

    def clear_cooldown(self, provider: str, key_index: int) -> None:
        """Remove a key from the cooldown set."""
        with self._lock:
            self._active_cooldowns.discard((provider, key_index))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return a snapshot of all tracked metrics.

        Returns:
            A plain ``dict`` that can be serialised to JSON for dashboards
            or diagnostic logs.
        """
        with self._lock:
            avg_latency = (
                self._total_latency / self._latency_count
                if self._latency_count > 0
                else 0.0
            )
            return {
                "requests_per_provider": dict(self._requests_per_provider),
                "requests_per_key": {
                    f"{p}:key_{k}": v
                    for (p, k), v in self._requests_per_key.items()
                },
                "success_count": self._success_count,
                "failure_count": self._failure_count,
                "average_latency_seconds": round(avg_latency, 4),
                "total_requests": self._success_count + self._failure_count,
                "current_retries": dict(self._current_retries),
                "active_cooldowns": [
                    {"provider": p, "key_index": k}
                    for p, k in sorted(self._active_cooldowns)
                ],
            }
