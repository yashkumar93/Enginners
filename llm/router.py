"""RouterLLM — the single LLM interface for every CrewAI agent.

This module is the heart of the multi-provider router.  ``RouterLLM``
subclasses ``crewai.llms.base_llm.BaseLLM`` so it can be dropped in
anywhere CrewAI expects an LLM — including the ``"llm"`` field in
JSON/JSONC agent configs via ``{"python": "llm.router.<attribute>"}``.

Architecture overview::

    Agent JSONC config
        └─ {"python": "llm.router.backend_engineer_llm"}
             └─ RouterLLM.call(messages, …)
                  └─ RetryHandler.execute_with_retry(…)
                       └─ BaseProvider.call_llm(…)
                            └─ crewai.LLM(model="groq/…", api_key="…").call(…)

Shared singletons (module-level):
    All ``RouterLLM`` instances reference the **same** pool, cooldown
    manager, logger, metrics, and retry handler.  This means cooldown
    state and metrics are consistent even when CrewAI runs agents in
    parallel threads.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, PrivateAttr

from crewai.llms.base_llm import BaseLLM

from llm.config import AGENT_MODEL_MAP, PROVIDERS
from llm.cooldown import CooldownManager
from llm.logger import RouterLogger
from llm.metrics import RouterMetrics
from llm.provider_factory import ProviderFactory
from llm.provider_pool import ProviderPool
from llm.retry import AllProvidersExhaustedError, RetryHandler

# =====================================================================
# Module-level shared singletons (initialised ONCE at import time)
# =====================================================================

_cooldown_manager = CooldownManager()
_logger = RouterLogger()
_metrics = RouterMetrics()

_providers = ProviderFactory.create_all_providers(
    providers_config=PROVIDERS,
    cooldown_manager=_cooldown_manager,
)
_provider_pool = ProviderPool(providers=_providers)

_retry_handler = RetryHandler(
    provider_pool=_provider_pool,
    cooldown_manager=_cooldown_manager,
    logger=_logger,
    metrics=_metrics,
)

_logger.info(
    f"LLM Router initialised — providers: "
    f"{', '.join(sorted(_providers))} | "
    f"agents mapped: {len(AGENT_MODEL_MAP)}"
)


# =====================================================================
# RouterLLM — the custom BaseLLM subclass
# =====================================================================


class RouterLLM(BaseLLM):
    """A lightweight BaseLLM adapter that routes calls through the shared pool.

    Each instance is identified by ``agent_name`` and carries no mutable
    state of its own — all coordination happens via the module-level
    singletons.

    Pydantic fields:
        model:      Always ``"router"`` (satisfies BaseLLM's required field).
        agent_name: The human-readable agent name (e.g. ``"Backend Engineer"``).
    """

    agent_name: str = Field(
        ...,
        description="Name of the CrewAI agent this LLM serves",
    )

    # References to singletons (excluded from Pydantic serialisation)
    _pool: ProviderPool = PrivateAttr(default=None)           # type: ignore[assignment]
    _retry: RetryHandler = PrivateAttr(default=None)          # type: ignore[assignment]
    _router_logger: RouterLogger = PrivateAttr(default=None)  # type: ignore[assignment]
    _router_metrics: RouterMetrics = PrivateAttr(default=None) # type: ignore[assignment]

    def model_post_init(self, __context: Any) -> None:
        """Wire up singleton references after Pydantic construction."""
        self._pool = _provider_pool
        self._retry = _retry_handler
        self._router_logger = _logger
        self._router_metrics = _metrics

    # ------------------------------------------------------------------
    # BaseLLM abstract method implementation
    # ------------------------------------------------------------------

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str | Any:
        """Route an LLM call through the retry handler.

        1. Look up the agent's preferred (provider, model) from config.
        2. Delegate to ``RetryHandler.execute_with_retry()``.
        3. Return the raw LLM response string.

        Falls back to the first provider in ``PROVIDER_PRIORITY`` if the
        agent isn't in ``AGENT_MODEL_MAP``.
        """
        mapping = AGENT_MODEL_MAP.get(self.agent_name)
        if mapping is None:
            self._router_logger.error(
                f"Agent '{self.agent_name}' not found in AGENT_MODEL_MAP — "
                "using first available provider"
            )
            # Graceful degradation: pick the first available provider
            provider_name = next(iter(_providers), "groq")
            provider = _providers.get(provider_name)
            model = provider.models[0] if provider and provider.models else "llama-3.3-70b-versatile"
        else:
            provider_name = mapping.provider
            model = mapping.model

        # Build kwargs for the underlying crewai.LLM.call()
        call_kwargs: dict[str, Any] = {}
        if tools is not None:
            call_kwargs["tools"] = tools
        if callbacks is not None:
            call_kwargs["callbacks"] = callbacks
        if available_functions is not None:
            call_kwargs["available_functions"] = available_functions
        if from_task is not None:
            call_kwargs["from_task"] = from_task
        if from_agent is not None:
            call_kwargs["from_agent"] = from_agent
        if response_model is not None:
            call_kwargs["response_model"] = response_model

        # Clean cache_breakpoint key from all messages to prevent provider errors (e.g. Groq)
        if isinstance(messages, list):
            cleaned_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    msg_copy = msg.copy()
                    msg_copy.pop("cache_breakpoint", None)
                    cleaned_messages.append(msg_copy)
                elif hasattr(msg, "additional_kwargs") and isinstance(msg.additional_kwargs, dict):
                    msg.additional_kwargs.pop("cache_breakpoint", None)
                    cleaned_messages.append(msg)
                else:
                    cleaned_messages.append(msg)
            messages = cleaned_messages

        return self._retry.execute_with_retry(
            provider_name=provider_name,
            model=model,
            agent_name=self.agent_name,
            messages=messages,
            **call_kwargs,
        )

    async def acall(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str | Any:
        """Async wrapper — delegates to the synchronous ``call()`` for now.

        A true async implementation can be added later by making
        ``BaseProvider.call_llm`` async-aware.
        """
        return self.call(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )


# =====================================================================
# Pre-built agent LLM instances (referenced from JSONC agent configs)
# =====================================================================
# Usage in agents/<name>.jsonc:
#   "llm": {"python": "llm.router.project_manager_llm"}

project_manager_llm = RouterLLM(agent_name="Project Manager", model="router")
software_architect_llm = RouterLLM(agent_name="Software Architect", model="router")
backend_engineer_llm = RouterLLM(agent_name="Backend Engineer", model="router")
frontend_engineer_llm = RouterLLM(agent_name="Frontend Engineer", model="router")
qa_engineer_llm = RouterLLM(agent_name="QA Engineer", model="router")
devops_engineer_llm = RouterLLM(agent_name="DevOps Engineer", model="router")


# =====================================================================
# Dynamic factory (for agents not pre-registered above)
# =====================================================================


def get_llm(agent_name: str) -> RouterLLM:
    """Create a ``RouterLLM`` for an arbitrary agent name.

    Useful for dynamically-created agents or test harnesses that don't
    want to use the pre-built module-level instances.

    Args:
        agent_name: Must match a key in ``AGENT_MODEL_MAP`` for
                    deterministic routing.  Unrecognised names still
                    work but will fall back to the first provider.

    Returns:
        A fresh ``RouterLLM`` instance wired to the shared singletons.
    """
    return RouterLLM(agent_name=agent_name, model="router")


# =====================================================================
# Convenience accessors for the shared singletons
# =====================================================================


def get_metrics_summary() -> dict[str, Any]:
    """Return a snapshot of all router metrics."""
    return _metrics.get_summary()


def get_provider_pool() -> ProviderPool:
    """Return the shared provider pool (for advanced introspection)."""
    return _provider_pool


# =====================================================================
# Crew Embedder Configuration (RAG)
# =====================================================================
# This configuration is referenced from crew.jsonc to configure RAG:
#   "embedder": {"python": "llm.router.crew_embedder"}

import os
gemini_key = None
if "gemini" in _providers:
    gemini_keys = _providers["gemini"].keys
    if gemini_keys:
        gemini_key = gemini_keys[0]
if not gemini_key:
    gemini_key = os.environ.get("GEMINI_API_KEY_1") or os.environ.get("GEMINI_API_KEY")

if gemini_key:
    crew_embedder: dict[str, Any] = {
        "provider": "google-generativeai",
        "config": {
            "model_name": "gemini-embedding-001",
            "api_key": gemini_key
        }
    }
elif os.environ.get("OPENAI_API_KEY"):
    crew_embedder = {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": os.environ.get("OPENAI_API_KEY")
        }
    }
else:
    crew_embedder = {
        "provider": "sentence-transformer",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }

