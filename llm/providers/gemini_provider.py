"""Google Gemini LLM provider implementation.

Uses the ``gemini/`` prefix expected by CrewAI's LiteLLM-based ``LLM``
class to route requests to Google's Gemini API.
"""

from __future__ import annotations

from crewai import LLM

from llm.providers.base_provider import BaseProvider


class GeminiProvider(BaseProvider):
    """Adapter for calling Google Gemini models via CrewAI."""

    def make_llm(self, model: str, api_key: str) -> LLM:
        """Create a Gemini-prefixed ``crewai.LLM``.

        Example model string produced: ``"gemini/gemini-2.5-flash"``.
        """
        return LLM(
            model=f"gemini/{model}",
            api_key=api_key,
        )
