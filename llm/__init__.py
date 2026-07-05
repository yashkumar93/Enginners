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

import logging

_log = logging.getLogger("llm.init")

# ---------------------------------------------------------------------------
# CrewAI Cache Monkey-Patch
# ---------------------------------------------------------------------------
# Fixes a known bug in CrewAI where 'cache_breakpoint': true is injected into
# message payloads. This causes Groq Exception BadRequestError because Groq's
# API does not support cache breakpoints in messages.
#
# Double-insurance approach:
# 1. This monkey-patch prevents the key from being ADDED at source.
# 2. RouterLLM.call() strips any remaining keys before sending to providers.
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
    _log.debug("Monkey-patched crewai.llms.cache.mark_cache_breakpoint successfully")
except ImportError:
    _log.warning(
        "Could not monkey-patch crewai.llms.cache — module not found. "
        "The router-level cache_breakpoint stripping in RouterLLM.call() "
        "will handle this as a fallback."
    )

# Also patch the strip_cache_breakpoint to be a no-op for safety
try:
    import crewai.llms.cache as _crewai_cache2
    _crewai_cache2.strip_cache_breakpoint = lambda msg: None
except (ImportError, AttributeError):
    pass

from llm.router import RouterLLM, get_llm, crew_embedder, get_metrics_summary, get_provider_pool

__all__ = [
    "RouterLLM",
    "get_llm",
    "crew_embedder",
    "get_metrics_summary",
    "get_provider_pool",
]
