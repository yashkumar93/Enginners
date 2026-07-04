"""Provider Pool — manages the collection of active LLM providers.

The pool is the central lookup point that the retry handler uses to
find the primary provider for an agent and to iterate over fallback
providers when the primary is exhausted.
"""

from __future__ import annotations

from llm.config import PROVIDER_PRIORITY
from llm.providers.base_provider import BaseProvider


class ProviderPool:
    """Holds all initialised providers and exposes lookup / fallback APIs.

    Attributes:
        _providers: Internal mapping of provider name → instance.
    """

    def __init__(self, providers: dict[str, BaseProvider]) -> None:
        self._providers = providers

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_provider(self, name: str) -> BaseProvider:
        """Return the provider with the given *name*.

        Raises:
            KeyError: If no provider with *name* has been initialised.
        """
        try:
            return self._providers[name]
        except KeyError:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise KeyError(
                f"Provider '{name}' not found. Available: {available}"
            ) from None

    def get_fallback_providers(
        self,
        exclude: str,
    ) -> list[BaseProvider]:
        """Return providers ordered by ``PROVIDER_PRIORITY``, excluding *exclude*.

        Providers that have zero available keys (all on cooldown) are
        still returned — the retry handler may need to wait for cooldowns
        to expire.
        """
        return [
            self._providers[name]
            for name in PROVIDER_PRIORITY
            if name != exclude and name in self._providers
        ]

    def is_provider_available(self, name: str) -> bool:
        """Check if at least one key for *name* is off cooldown."""
        if name not in self._providers:
            return False
        return self._providers[name].get_available_key_count() > 0

    @property
    def provider_names(self) -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._providers)
