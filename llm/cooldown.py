"""Thread-safe cooldown management for rate-limited API keys.

When a key triggers a rate-limit (HTTP 429) or similar transient error,
it is "marked" with a timestamp.  Subsequent calls to ``is_on_cooldown``
will return ``True`` until the configured cooldown window expires,
ensuring the router skips the key and tries alternatives instead.
"""

from __future__ import annotations

import threading
import time

from llm.config import COOLDOWN_SECONDS


class CooldownManager:
    """Manages per-key cooldown windows in a thread-safe way.

    Internal state is a mapping of ``(provider, key_index)`` → ``float``
    (epoch timestamp when cooldown was triggered).  A ``threading.Lock``
    protects all reads and writes.
    """

    def __init__(self, cooldown_seconds: float = COOLDOWN_SECONDS) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        # (provider, key_index) → epoch timestamp of when cooldown started
        self._cooldowns: dict[tuple[str, int], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark_cooldown(self, provider: str, key_index: int) -> None:
        """Put the given key on cooldown starting *now*.

        Args:
            provider:  Provider name (e.g. ``"groq"``).
            key_index: Zero-based index into the provider's key list.
        """
        with self._lock:
            self._cooldowns[(provider, key_index)] = time.monotonic()

    def is_on_cooldown(self, provider: str, key_index: int) -> bool:
        """Check whether the key is still within its cooldown window.

        Returns ``False`` if the key was never marked or the window has
        already elapsed.
        """
        with self._lock:
            start = self._cooldowns.get((provider, key_index))
            if start is None:
                return False
            elapsed = time.monotonic() - start
            if elapsed >= self._cooldown_seconds:
                # Expired — auto-clear so future lookups are fast
                del self._cooldowns[(provider, key_index)]
                return False
            return True

    def get_remaining(self, provider: str, key_index: int) -> float:
        """Return seconds remaining in the cooldown window.

        Returns ``0.0`` if the key is not on cooldown.
        """
        with self._lock:
            start = self._cooldowns.get((provider, key_index))
            if start is None:
                return 0.0
            remaining = self._cooldown_seconds - (time.monotonic() - start)
            return max(remaining, 0.0)

    def clear(self, provider: str, key_index: int) -> None:
        """Manually clear cooldown for a specific key.

        Useful for operator overrides or test fixtures.
        """
        with self._lock:
            self._cooldowns.pop((provider, key_index), None)

    def clear_all(self) -> None:
        """Clear all active cooldowns."""
        with self._lock:
            self._cooldowns.clear()
