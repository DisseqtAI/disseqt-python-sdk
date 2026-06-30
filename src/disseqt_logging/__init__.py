"""Shared structured logging for the Disseqt Python SDK.

A standard-library-only logger (no third-party dependencies) that gives the SDK
production-grade structured logs: JSON or console output, automatic
``service``/``env``/``host`` base fields, PII/credential redaction, a
privacy-safe payload :func:`digest`, an error envelope, and a dynamic level.

Both SDK sub-packages (``disseqt_sdk`` and ``disseqt_agentic_sdk``) build on this
shared package, so the whole SDK emits one consistent log schema.

Typical use::

    from disseqt_logging import get_logger, configure

    configure(level="info")                 # optional; env vars also work
    log = get_logger(__name__)
    log.info("validation.response", status=200, latency_ms=12.3)
"""

from __future__ import annotations

from disseqt_logging.config import LoggerConfig, level_name, parse_level
from disseqt_logging.logger import (
    Logger,
    configure,
    current_level,
    debug,
    digest,
    disable,
    error,
    get_logger,
    info,
    set_level,
    stdlib_logger,
    warn,
)
from disseqt_logging.redaction import redact_field, redact_string, sensitive_key

__all__ = [
    "Logger",
    "LoggerConfig",
    "configure",
    "current_level",
    "debug",
    "digest",
    "disable",
    "error",
    "get_logger",
    "info",
    "level_name",
    "parse_level",
    "redact_field",
    "redact_string",
    "sensitive_key",
    "set_level",
    "stdlib_logger",
    "warn",
]
