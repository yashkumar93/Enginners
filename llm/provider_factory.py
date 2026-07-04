"""Factory for creating LLM provider instances.

Centralises the mapping from provider names to their concrete classes,
so the rest of the router never has to ``import`` provider modules
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm.providers.base_provider import BaseProvider
from llm.providers.gemini_provider import GeminiProvider
from llm.providers.groq_provider import GroqProvider

if TYPE_CHECKING:
    from llm.config import ProviderConfig
    from llm.cooldown import CooldownManager


# Registry of provider name → class
_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}


class ProviderFactory:
    """Creates ``BaseProvider`` instances from configuration data."""

    @staticmethod
    def create_provider(
        name: str,
        keys: list[str],
        models: list[str],
        cooldown_manager: CooldownManager,
    ) -> BaseProvider:
        """Instantiate a single provider by name.

        Args:
            name:             Provider identifier (must exist in the registry).
            keys:             List of API keys for this provider.
            models:           List of supported model identifiers.
            cooldown_manager: Shared cooldown tracker.

        Returns:
            A concrete ``BaseProvider`` subclass ready for use.

        Raises:
            ValueError: If *name* is not a registered provider.
        """
        cls = _PROVIDER_CLASSES.get(name)
        if cls is None:
            registered = ", ".join(sorted(_PROVIDER_CLASSES))
            raise ValueError(
                f"Unknown provider '{name}'. "
                f"Registered providers: {registered}"
            )
        return cls(
            name=name,
            keys=keys,
            models=models,
            cooldown_manager=cooldown_manager,
        )

    @staticmethod
    def create_all_providers(
        providers_config: dict[str, ProviderConfig],
        cooldown_manager: CooldownManager,
    ) -> dict[str, BaseProvider]:
        """Bulk-create providers from the full config mapping.

        Providers with **no** valid API keys are silently skipped —
        they will never be reachable at runtime anyway.

        Returns:
            A ``dict`` of ``provider_name → BaseProvider`` instances.
        """
        result: dict[str, BaseProvider] = {}
        for name, cfg in providers_config.items():
            if not cfg.keys:
                continue  # no keys → skip
            result[name] = ProviderFactory.create_provider(
                name=name,
                keys=cfg.keys,
                models=cfg.models,
                cooldown_manager=cooldown_manager,
            )
        return result
