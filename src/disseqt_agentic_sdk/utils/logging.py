"""Structured logging utilities for the agentic SDK.

Thin wrapper over the SDK-wide :mod:`disseqt_logging` package so the agentic and
validation SDKs emit one consistent, redacted, structured log schema.

``get_logger`` returns a standard-library :class:`logging.Logger` (unchanged
from earlier releases) whose records are rendered through the shared structured
handler. Every stdlib method works (``setLevel`` / ``addHandler`` /
``isEnabledFor`` / ``isinstance(..., logging.Logger)`` / …); pass structured
fields via ``extra={...}``. ``set_log_level`` is unchanged.
"""

from __future__ import annotations

import logging

from disseqt_logging import set_level, stdlib_logger

# Default logger name (kept for backward compatibility).
DEFAULT_LOGGER_NAME = "disseqt_agentic_sdk"


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a structured logger instance.

    Args:
        name: Logger name (defaults to the agentic SDK logger name).

    Returns:
        A standard-library :class:`logging.Logger`. Its output (JSON or
        console), base fields, and PII redaction are governed by
        ``disseqt_logging``; the SDK is silent until logging is enabled via
        :func:`set_log_level` / ``disseqt_logging.configure`` / the
        ``DISSEQT_LOG_LEVEL`` environment variable.
    """
    return stdlib_logger(name or DEFAULT_LOGGER_NAME)


def set_log_level(level: str | int) -> None:
    """Set the SDK-wide log level (and enable output if it was silent).

    Args:
        level: Level name ("debug"/"info"/"warn"/"error") or numeric level.
            Unrecognized values fall back to INFO.
    """
    set_level(level)
