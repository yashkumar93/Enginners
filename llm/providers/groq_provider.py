"""Groq LLM provider implementation.

Uses the ``groq/`` prefix expected by CrewAI's LiteLLM-based ``LLM``
class to route requests to Groq's inference API.
"""

from __future__ import annotations

from crewai import LLM

from llm.providers.base_provider import BaseProvider


class GroqProvider(BaseProvider):
    """Adapter for calling Groq-hosted models via CrewAI."""

    def make_llm(self, model: str, api_key: str) -> LLM:
        """Create a Groq-prefixed ``crewai.LLM``.

        Example model string produced: ``"groq/llama-3.3-70b-versatile"``.
        """
        return LLM(
            model=f"groq/{model}",
            api_key=api_key,
        )
