"""LLM Router Package.

Provides a multi-provider LLM routing layer for CrewAI agents.
The router handles API key rotation, rate-limit cooldowns, cross-provider
failover, structured logging, and per-request metrics — all behind a
single ``RouterLLM`` class that plugs into CrewAI's ``BaseLLM`` interface.

Quick start (JSONC agent config)::

    "llm": {"python": "llm.router.backend_engineer_llm"}

Programmatic usage::

    from llm import RouterLLM, get_llm

    llm = get_llm("Backend Engineer")
    response = llm.call("Write a REST API for user management.")
"""

# ---------------------------------------------------------------------------
# CrewAI Cache Monkey-Patch
# ---------------------------------------------------------------------------
# Fixes a known bug in CrewAI where 'cache_breakpoint': true is injected into
# message payloads. This causes Groq Exception BadRequestError because Groq's
# API does not support cache breakpoints in messages.
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except ImportError:
    pass

from llm.router import RouterLLM, get_llm, crew_embedder

__all__ = [
    "RouterLLM",
    "get_llm",
    "crew_embedder",
]
