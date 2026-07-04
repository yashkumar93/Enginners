"""Structured logging for the LLM Router.

Every LLM request is logged with:
    Timestamp · Agent Name · Provider · API Key Index · Model ·
    Latency (seconds) · Retry Count · Success (bool) · Failure Reason

Output goes to both the console (via the ``logging`` stdlib) and a
persistent file at ``<project_root>/llm_router.log``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from llm.config import LOG_FILE_PATH


class RouterLogger:
    """Thread-safe structured logger for router events.

    Uses Python's built-in ``logging`` module under the hood, which is
    already thread-safe.  Two handlers are attached:

    * ``StreamHandler`` — prints to stderr / console
    * ``FileHandler``   — appends to ``llm_router.log``
    """

    _FORMAT: str = (
        "[%(levelname)s] %(asctime)s | %(message)s"
    )

    def __init__(self, log_file: Path = LOG_FILE_PATH) -> None:
        self._logger = logging.getLogger("llm_router")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # avoid duplicate root-logger output

        # Only attach handlers once (guards against repeated singleton init)
        if not self._logger.handlers:
            formatter = logging.Formatter(self._FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

            # Console handler
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            self._logger.addHandler(console)

            # File handler
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_request(
        self,
        *,
        agent_name: str,
        provider: str,
        key_index: int,
        model: str,
        latency: float,
        retry_count: int,
        success: bool,
        failure_reason: str | None = None,
    ) -> None:
        """Emit a single structured log line for an LLM request."""
        status = "Success" if success else f"Failed ({failure_reason})"
        msg = (
            f"Agent: {agent_name} | "
            f"Provider: {provider.capitalize()} | "
            f"Key: {key_index} | "
            f"Model: {model} | "
            f"Latency: {latency:.2f}s | "
            f"Retries: {retry_count} | "
            f"Status: {status}"
        )
        if success:
            self._logger.info(msg)
        else:
            self._logger.warning(msg)

    def log_cooldown_activated(
        self,
        provider: str,
        key_index: int,
        reason: str,
    ) -> None:
        """Log when an API key enters cooldown."""
        self._logger.warning(
            f"Cooldown activated | Provider: {provider.capitalize()} | "
            f"Key: {key_index} | Reason: {reason}"
        )

    def log_provider_switch(
        self,
        from_provider: str,
        to_provider: str,
        agent_name: str,
    ) -> None:
        """Log a fallback to a different provider."""
        self._logger.info(
            f"Provider switch | Agent: {agent_name} | "
            f"From: {from_provider.capitalize()} → To: {to_provider.capitalize()}"
        )

    def log_all_exhausted(self, agent_name: str) -> None:
        """Log when all providers and keys have been exhausted."""
        self._logger.error(
            f"All providers exhausted | Agent: {agent_name} | "
            "No available keys across any provider"
        )

    def debug(self, msg: str) -> None:
        """Pass-through to underlying DEBUG logger."""
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        """Pass-through to underlying INFO logger."""
        self._logger.info(msg)

    def error(self, msg: str) -> None:
        """Pass-through to underlying ERROR logger."""
        self._logger.error(msg)
