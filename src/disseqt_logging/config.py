"""Logger construction configuration and level helpers."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, TextIO

# Canonical level names accepted everywhere, mapped to stdlib numeric levels.
_NAME_TO_LEVEL: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "fatal": logging.CRITICAL,
}

_LEVEL_TO_NAME: dict[int, str] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "fatal",
}


def parse_level(level: str | int) -> int:
    """Coerce a level name or numeric level to a stdlib numeric level.

    Empty / unrecognized names fall back to ``INFO`` rather than raising, so a
    bad env var never breaks logging.

    Args:
        level: A canonical level name (case-insensitive) or numeric level.

    Returns:
        The corresponding :mod:`logging` numeric level.
    """
    if isinstance(level, int):
        return level
    try:
        return _NAME_TO_LEVEL.get(level.strip().lower(), logging.INFO)
    except AttributeError:
        # Defensive: a non-str/int slipped past the type hint at runtime.
        return logging.INFO


def level_name(level: int) -> str:
    """Return the canonical lowercase name for a numeric level."""
    return _LEVEL_TO_NAME.get(level, "info")


def _truthy(value: str) -> bool:
    """Interpret an env-var string as a boolean (default-ish to True)."""
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass
class LoggerConfig:
    """Configuration for the shared logger.

    Every field has a sane default so ``LoggerConfig()`` yields a usable,
    production-safe logger (JSON to stderr at info level with redaction on).

    Attributes:
        level: Minimum level emitted ("debug"/"info"/"warn"/"error").
        service: Value bound to the ``service`` field on every line.
        env: Value bound to the ``env`` field on every line.
        version: Optional build/commit value bound to ``version`` when set.
        component: Optional component name bound to ``component`` when set.
        fmt: Renderer — ``"json"``, ``"console"``, or ``"auto"`` (console when
            the stream is a TTY, JSON otherwise).
        redact: When True, run PII/credential redaction over string fields.
        stream: The output stream (defaults to ``sys.stderr``).
        file_path: When set, also tee output to a size-rotated file.
        file_max_bytes: Rotation threshold for the file handler.
        file_backups: Number of rotated files to keep.
    """

    level: str | int = "info"
    service: str = "disseqt-ai-sdk"
    env: str = "production"
    version: str = ""
    component: str = ""
    fmt: str = "auto"
    redact: bool = True
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    file_path: str | None = None
    file_max_bytes: int = 100 * 1024 * 1024
    file_backups: int = 7

    @classmethod
    def from_env(cls, **overrides: Any) -> LoggerConfig:
        """Build a config from ``DISSEQT_*`` env vars, then apply ``overrides``.

        Recognized env vars: ``DISSEQT_LOG_LEVEL``, ``DISSEQT_LOG_SERVICE``,
        ``DISSEQT_ENV``, ``DISSEQT_LOG_FORMAT`` (``json``/``console``/``auto``),
        ``DISSEQT_LOG_REDACT`` (``0`` disables), ``DISSEQT_LOG_FILE``.

        Args:
            **overrides: Explicit values that take precedence over env vars.

        Returns:
            A populated :class:`LoggerConfig`.
        """
        env = os.environ
        cfg = cls(
            level=env.get("DISSEQT_LOG_LEVEL", "info") or "info",
            service=env.get("DISSEQT_LOG_SERVICE", "disseqt-ai-sdk") or "disseqt-ai-sdk",
            env=env.get("DISSEQT_ENV", "production") or "production",
            version=env.get("DISSEQT_LOG_VERSION", ""),
            fmt=env.get("DISSEQT_LOG_FORMAT", "auto") or "auto",
            redact=_truthy(env.get("DISSEQT_LOG_REDACT", "1")),
            file_path=env.get("DISSEQT_LOG_FILE") or None,
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def resolve_format(self) -> str:
        """Resolve ``fmt`` ("auto") to a concrete ``"json"`` or ``"console"``."""
        if self.fmt == "console":
            return "console"
        if self.fmt == "json":
            return "json"
        # auto: human-friendly console when attached to a terminal, else JSON.
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        return "console" if is_tty else "json"
