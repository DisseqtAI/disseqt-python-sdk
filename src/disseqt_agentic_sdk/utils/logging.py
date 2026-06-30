"""Structured logging utilities for the agentic SDK.

Thin wrapper over the SDK-wide :mod:`disseqt_logging` package so the agentic and
validation SDKs emit one consistent, redacted, structured log schema. The public
``get_logger`` / ``set_log_level`` surface is unchanged; ``get_logger`` now
returns a :class:`disseqt_logging.Logger`, whose ``debug`` / ``info`` /
``warning`` / ``error`` methods are drop-in for the existing call sites.
"""

from __future__ import annotations

from disseqt_logging import Logger, set_level
from disseqt_logging import get_logger as _get_logger

# Default logger name (kept for backward compatibility).
DEFAULT_LOGGER_NAME = "disseqt_agentic_sdk"


def get_logger(name: str | None = None) -> Logger:
    """Get a structured logger instance.

    Args:
        name: Logger name (defaults to the agentic SDK logger name).

    Returns:
        A :class:`disseqt_logging.Logger`. Its output (JSON or console),
        base fields, and PII redaction are governed by ``disseqt_logging``.
    """
    return _get_logger(name or DEFAULT_LOGGER_NAME)


def set_log_level(level: str | int) -> None:
    """Set the SDK-wide log level.

    Args:
        level: Level name ("debug"/"info"/"warn"/"error") or numeric level.
            Unrecognized values fall back to INFO.
    """
    set_level(level)
