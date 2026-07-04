"""LLM Router Configuration.

Loads API keys from environment variables and defines the mapping between
agents and their preferred LLM providers/models. All secrets come from
the environment — nothing is hardcoded.

Environment variables consumed:
    GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3
    GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from the project root (two levels up from this file)
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_ENV_PATH: Final[Path] = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# Resolve environment variables and prevent Pydantic Settings overrides
# ---------------------------------------------------------------------------
# Prevents MODEL=groq/llama-3.3-70b-versatile in .env from overriding the
# Google embedding model name due to Pydantic Settings validation_alias choices.
os.environ["EMBEDDINGS_GOOGLE_GENERATIVE_AI_MODEL_NAME"] = "gemini-embedding-001"

# Automatically map GEMINI_API_KEY to the first active Gemini API key if unset
if not os.environ.get("GEMINI_API_KEY"):
    _gemini_key = os.environ.get("GEMINI_API_KEY_1", "").strip()
    if _gemini_key:
        os.environ["GEMINI_API_KEY"] = _gemini_key


# ---------------------------------------------------------------------------
# Helper: collect keys from env, filtering out empty / unset values
# ---------------------------------------------------------------------------
def _load_keys(prefix: str, count: int = 3) -> list[str]:
    """Read ``{prefix}_1`` … ``{prefix}_{count}`` from the environment.

    Only non-empty strings are included so the provider never attempts
    a call with a blank key.
    """
    keys: list[str] = []
    for i in range(1, count + 1):
        value = os.environ.get(f"{prefix}_{i}", "").strip()
        if value:
            keys.append(value)
    return keys


# ---------------------------------------------------------------------------
# Data classes for structured config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderConfig:
    """Immutable configuration for a single LLM provider."""

    keys: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentModelMapping:
    """Maps an agent to its preferred provider and model."""

    provider: str
    model: str


# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------
PROVIDERS: Final[dict[str, ProviderConfig]] = {
    "groq": ProviderConfig(
        keys=_load_keys("GROQ_API_KEY"),
        models=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
    ),
    "gemini": ProviderConfig(
        keys=_load_keys("GEMINI_API_KEY"),
        models=[
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
    ),
}

# ---------------------------------------------------------------------------
# Agent → (provider, model) mapping (supports environment overrides)
# ---------------------------------------------------------------------------
AGENT_MODEL_MAP: Final[dict[str, AgentModelMapping]] = {
    "Project Manager":     AgentModelMapping(
        provider=os.environ.get("PROJECT_MANAGER_PROVIDER", "groq"),
        model=os.environ.get("PROJECT_MANAGER_MODEL", "llama-3.1-8b-instant")
    ),
    "Software Architect":  AgentModelMapping(
        provider=os.environ.get("SOFTWARE_ARCHITECT_PROVIDER", "groq"),
        model=os.environ.get("SOFTWARE_ARCHITECT_MODEL", "llama-3.1-8b-instant")
    ),
    "Backend Engineer":    AgentModelMapping(
        provider=os.environ.get("BACKEND_ENGINEER_PROVIDER", "groq"),
        model=os.environ.get("BACKEND_ENGINEER_MODEL", "llama-3.1-8b-instant")
    ),
    "Frontend Engineer":   AgentModelMapping(
        provider=os.environ.get("FRONTEND_ENGINEER_PROVIDER", "gemini"),
        model=os.environ.get("FRONTEND_ENGINEER_MODEL", "gemini-2.5-flash")
    ),
    "QA Engineer":         AgentModelMapping(
        provider=os.environ.get("QA_ENGINEER_PROVIDER", "groq"),
        model=os.environ.get("QA_ENGINEER_MODEL", "llama-3.1-8b-instant")
    ),
    "DevOps Engineer":     AgentModelMapping(
        provider=os.environ.get("DEVOPS_ENGINEER_PROVIDER", "gemini"),
        model=os.environ.get("DEVOPS_ENGINEER_MODEL", "gemini-2.5-flash")
    ),
}

# ---------------------------------------------------------------------------
# Retry / resilience settings (overridable via env vars)
# ---------------------------------------------------------------------------
MAX_RETRIES: Final[int] = int(os.environ.get("LLM_ROUTER_MAX_RETRIES", "6"))
COOLDOWN_SECONDS: Final[float] = float(
    os.environ.get("LLM_ROUTER_COOLDOWN_SECONDS", "60")
)

# ---------------------------------------------------------------------------
# Fallback priority order — tried in sequence when the primary provider fails
# ---------------------------------------------------------------------------
PROVIDER_PRIORITY: Final[list[str]] = ["groq", "gemini"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOG_FILE_PATH: Final[Path] = _PROJECT_ROOT / "llm_router.log"
