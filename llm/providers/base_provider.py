"""Abstract base class for LLM providers.

Each concrete provider (Groq, Gemini, …) subclasses ``BaseProvider`` and
overrides ``make_llm`` to construct a ``crewai.LLM`` with the correct
model-string prefix and API key.

Key rotation is handled via round-robin with cooldown awareness: the
``get_next_key`` method cycles through available keys, automatically
skipping any that are currently on cooldown.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from crewai import LLM

if TYPE_CHECKING:
    from llm.cooldown import CooldownManager


class BaseProvider(ABC):
    """Provider-agnostic adapter for making LLM calls with key rotation.

    Attributes:
        name:      Unique provider identifier (e.g. ``"groq"``).
        keys:      List of API keys available for this provider.
        models:    List of model identifiers supported by this provider.
    """

    def __init__(
        self,
        name: str,
        keys: list[str],
        models: list[str],
        cooldown_manager: CooldownManager,
    ) -> None:
        self.name = name
        self.keys = keys
        self.models = models
        self._cooldown_manager = cooldown_manager
        self._current_index: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Key rotation
    # ------------------------------------------------------------------

    def get_next_key(self) -> tuple[int, str]:
        """Return ``(key_index, api_key)`` for the next available key.

        Uses round-robin traversal, skipping any key currently on
        cooldown.  If **all** keys are on cooldown, raises
        ``RuntimeError``.
        """
        with self._lock:
            total = len(self.keys)
            if total == 0:
                raise RuntimeError(
                    f"Provider '{self.name}' has no API keys configured"
                )

            for _ in range(total):
                idx = self._current_index % total
                self._current_index = (self._current_index + 1) % total

                if not self._cooldown_manager.is_on_cooldown(self.name, idx):
                    return idx, self.keys[idx]

            raise RuntimeError(
                f"All API keys for provider '{self.name}' are on cooldown"
            )

    def get_available_key_count(self) -> int:
        """Return the number of keys NOT currently on cooldown."""
        return sum(
            1
            for idx in range(len(self.keys))
            if not self._cooldown_manager.is_on_cooldown(self.name, idx)
        )

    # ------------------------------------------------------------------
    # LLM construction & invocation
    # ------------------------------------------------------------------

    @abstractmethod
    def make_llm(self, model: str, api_key: str) -> LLM:
        """Create a ``crewai.LLM`` instance for the given model and key.

        Subclasses must implement this to apply the correct provider
        prefix (e.g. ``"groq/…"``, ``"gemini/…"``).
        """

    def call_llm(
        self,
        model: str,
        key_index: int,
        messages: Any,
        **kwargs: Any,
    ) -> str | Any:
        """Instantiate a fresh ``crewai.LLM`` and forward the call.

        The LLM is created per-call (not cached) so each invocation
        uses the explicitly selected API key, avoiding stale-key issues.

        Args:
            model:     Model identifier (without provider prefix).
            key_index: Index into ``self.keys`` for the API key to use.
            messages:  Messages payload forwarded to ``LLM.call()``.
            **kwargs:  Additional keyword arguments forwarded verbatim.

        Returns:
            The string (or structured) response from the LLM.
        """
        api_key = self.keys[key_index]
        llm = self.make_llm(model=model, api_key=api_key)
        return llm.call(messages=messages, **kwargs)
