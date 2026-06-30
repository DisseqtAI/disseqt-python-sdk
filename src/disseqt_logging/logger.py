"""The shared structured logger: ``Logger`` class, formatter, and facade.

A self-contained logger built on the standard library that matches the Disseqt
platform logger's output and feature set:

* structured JSON (or human console) output with ISO-8601 UTC timestamps,
* always-on base fields (``service`` / ``env`` / ``host`` + optional
  ``version`` / ``component``),
* PII / credential redaction over string fields (see :mod:`.redaction`),
* an error envelope (``error`` / ``error_type`` / ``error_code``) and optional
  traceback on :meth:`Logger.error`,
* a privacy-safe :func:`digest` for payload correlation,
* a dynamic level (:func:`set_level`) and ``bind`` / ``with_component`` sub-loggers.

Internally every SDK logger is reparented under a single ``disseqt`` root logger
so one handler serves the whole SDK; the synthetic prefix is stripped from the
emitted ``logger`` field.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import sys
import threading
import traceback
from datetime import datetime, timezone
from typing import Any

from disseqt_logging.config import LoggerConfig, level_name, parse_level
from disseqt_logging.redaction import redact_field

# A distribution-specific root so we never capture or block a client's own
# loggers. A short name like "disseqt" would hijack any "disseqt.*" logger the
# app uses (propagate=False would swallow it); "disseqt_ai_sdk" is unique to this
# package. The prefix is stripped from the emitted ``logger`` field.
_ROOT_NAME = "disseqt_ai_sdk"
_RECORD_KEY = "_disseqt"
_OWNED_ATTR = "_disseqt_owned"

# Always-set line metadata; a same-named caller field is renamed (``field_<k>``)
# rather than allowed to clobber it. Excluded on purpose: the error-envelope
# keys (set conditionally — they should overwrite a caller field) and the
# optional ``version`` / ``component`` keys (user-settable via ``bind`` /
# ``with_component``, so a caller value should land as-is).
_STRUCTURAL: frozenset[str] = frozenset(
    {
        "timestamp",
        "level",
        "event",
        "service",
        "env",
        "host",
        "logger",
    }
)

# Standard LogRecord attributes; anything else on a plain stdlib record is a
# caller-supplied ``extra=`` field that should flow into the structured output.
_RESERVED_RECORD_ATTRS: frozenset[str] = frozenset(vars(logging.makeLogRecord({})).keys()) | {
    "message",
    "asctime",
    "taskName",
}


def _stdlib_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Extract caller ``extra=`` fields from a plain stdlib log record."""
    return {
        key: value
        for key, value in vars(record).items()
        if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_")
    }


_lock = threading.RLock()
_config: LoggerConfig | None = None
_configured = False  # handlers installed (real or silent NullHandler)
_active = False  # real output handler installed (vs the silent NullHandler)
_default_logger: Logger | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _Digest:
    """A privacy-safe payload digest the redactor must never re-process.

    Carried as a distinct (non-``str``) type so the field redactor skips it by
    type rather than re-scanning its text — otherwise a digit run inside the
    SHA-256 hex would be mis-detected as a phone number and corrupted. Renders as
    the plain ``"len=<n> sha256=<16hex>"`` text in the final log line and
    compares equal to that string.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        """Wrap the rendered digest ``text``."""
        self.text = text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Digest({self.text!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Digest):
            return self.text == other.text
        if isinstance(other, str):
            return self.text == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.text)


def digest(value: str | bytes) -> _Digest:
    """Return a privacy-safe ``"len=<n> sha256=<16hex>"`` summary of ``value``.

    Lets payloads (which may contain user prompts) be correlated in logs without
    ever exposing their content. String length is the character count; bytes
    length is the byte count; the hash is over the UTF-8 / raw bytes either way.

    Args:
        value: The string or bytes to summarize.

    Returns:
        A :class:`_Digest` token that renders as the digest text and is exempt
        from redaction. Use ``str(...)`` for the plain string. ``"len=0"`` for
        empty input.
    """
    if isinstance(value, str):
        raw = value.encode("utf-8")
        length = len(value)
    else:
        raw = bytes(value)
        length = len(value)
    if length == 0:
        return _Digest("len=0")
    return _Digest(f"len={length} sha256={hashlib.sha256(raw).hexdigest()[:16]}")


def _iso(created: float) -> str:
    """Render a record timestamp as ISO-8601 UTC with a ``Z`` suffix."""
    return datetime.fromtimestamp(created, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _exc_from(exc_info: Any) -> BaseException | None:
    """Resolve an ``exc_info`` argument (``True`` / exception / 3-tuple) to an exception."""
    if exc_info is True:
        return sys.exc_info()[1]
    if isinstance(exc_info, BaseException):
        return exc_info
    if isinstance(exc_info, tuple) and len(exc_info) == 3:
        candidate = exc_info[1]
        return candidate if isinstance(candidate, BaseException) else None
    return None


def _format_exc(exc_info: Any) -> str | None:
    """Render a traceback string from an ``exc_info`` argument, or ``None``."""
    if exc_info is True:
        exc_type, exc_value, exc_tb = sys.exc_info()
    elif isinstance(exc_info, BaseException):
        exc_type, exc_value, exc_tb = type(exc_info), exc_info, exc_info.__traceback__
    elif isinstance(exc_info, tuple) and len(exc_info) == 3:
        exc_type, exc_value, exc_tb = exc_info
    else:
        return None
    if exc_value is None:
        return None
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip()


def _error_envelope(err: BaseException | None, exc_info: Any) -> dict[str, Any] | None:
    """Build the ``error`` / ``error_type`` / ``error_code`` envelope, or ``None``."""
    exc = err if err is not None else _exc_from(exc_info)
    if exc is None:
        return None
    out: dict[str, Any] = {"error": str(exc), "error_type": type(exc).__name__}
    code = getattr(exc, "code", None)
    if code is not None:
        value = getattr(code, "value", None)
        out["error_code"] = value if isinstance(value, str) else str(code)
    return out


# --------------------------------------------------------------------------- #
# Formatter
# --------------------------------------------------------------------------- #


class _DisseqtFormatter(logging.Formatter):
    """Render a record as a redacted JSON or console line."""

    def __init__(self, config: LoggerConfig) -> None:
        """Capture the config (base fields, format mode, redaction toggle)."""
        super().__init__()
        self._config = config
        self._mode = config.resolve_format()
        self._redact = config.redact
        self._host = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        """Assemble, redact, and render ``record``."""
        payload: dict[str, Any] | None = getattr(record, _RECORD_KEY, None)
        if payload is not None:
            # Record emitted via the :class:`Logger` wrapper.
            fields: dict[str, Any] = payload.get("fields") or {}
            envelope: dict[str, Any] | None = payload.get("error")
            exc_text: str | None = payload.get("exc_text")
        else:
            # Plain stdlib record (e.g. via :func:`stdlib_logger` or any
            # ``logging.getLogger`` child under our root). Pull ``extra=`` fields
            # off the record and build the envelope from ``exc_info``.
            fields = _stdlib_fields(record)
            exc = record.exc_info[1] if record.exc_info else None
            envelope = _error_envelope(exc, None) if exc is not None else None
            exc_text = self.formatException(record.exc_info) if record.exc_info else None

        logger_name = record.name
        if logger_name.startswith(_ROOT_NAME + "."):
            logger_name = logger_name[len(_ROOT_NAME) + 1 :]

        out: dict[str, Any] = {
            "timestamp": _iso(record.created),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
            "service": self._config.service,
            "env": self._config.env,
            "host": self._host,
            "logger": logger_name,
        }
        if self._config.version:
            out["version"] = self._config.version
        if self._config.component:
            out["component"] = self._config.component

        for key, value in fields.items():
            safe_key = f"field_{key}" if key in _STRUCTURAL else key
            out[safe_key] = redact_field(safe_key, value) if self._redact else value

        if envelope:
            for key, value in envelope.items():
                out[key] = redact_field(key, value) if self._redact else value
        if exc_text:
            # Rendered last and left un-redacted, matching the platform logger.
            out["exception"] = exc_text

        if self._mode == "console":
            return self._render_console(out)
        return json.dumps(out, default=str, sort_keys=True)

    @staticmethod
    def _render_console(out: dict[str, Any]) -> str:
        """Render a human-friendly single line (plus traceback) for dev use."""
        out = dict(out)
        timestamp = out.pop("timestamp")
        level = out.pop("level").upper()
        event = out.pop("event")
        exc = out.pop("exception", None)
        for noisy in ("service", "env", "host", "version", "component", "logger"):
            out.pop(noisy, None)
        head = f"{timestamp} [{level}] {event}"
        if out:
            head += "  " + " ".join(f"{k}={out[k]}" for k in sorted(out))
        if exc:
            head += "\n" + exc
        return head


# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #


class Logger:
    """A structured logger wrapping a stdlib logger with kit conventions.

    Obtain one via :func:`get_logger`. Emit methods take a message plus
    arbitrary keyword fields; for stdlib-call-site compatibility they also accept
    an ``extra`` mapping (merged into the fields) and an ``exc_info`` argument.
    ``error`` additionally accepts a positional exception and stamps the error
    envelope.
    """

    __slots__ = ("_log", "_bound")

    def __init__(self, stdlib_logger: logging.Logger, bound: dict[str, Any] | None = None) -> None:
        """Wrap ``stdlib_logger`` with optional pre-``bound`` fields."""
        self._log = stdlib_logger
        self._bound = dict(bound or {})

    def debug(self, msg: str, **fields: Any) -> None:
        """Log at debug level."""
        self._emit(logging.DEBUG, msg, None, fields)

    def info(self, msg: str, **fields: Any) -> None:
        """Log at info level."""
        self._emit(logging.INFO, msg, None, fields)

    def warn(self, msg: str, **fields: Any) -> None:
        """Log at warning level."""
        self._emit(logging.WARNING, msg, None, fields)

    # Alias so stdlib-style ``logger.warning(...)`` call sites keep working.
    warning = warn

    def error(self, msg: str, err: BaseException | None = None, **fields: Any) -> None:
        """Log at error level, stamping the error envelope from ``err`` / ``exc_info``."""
        self._emit(logging.ERROR, msg, err, fields)

    def _emit(
        self, level: int, msg: str, err: BaseException | None, fields: dict[str, Any]
    ) -> None:
        """Assemble fields and write the entry (short-circuits when level-disabled)."""
        if not self._log.isEnabledFor(level):
            return
        extra = fields.pop("extra", None)
        exc_info = fields.pop("exc_info", None)
        merged: dict[str, Any] = dict(self._bound)
        if isinstance(extra, dict):
            merged.update(extra)
        merged.update(fields)
        payload = {
            "fields": merged,
            "error": _error_envelope(err, exc_info),
            "exc_text": _format_exc(exc_info) if exc_info else None,
        }
        self._log.log(level, msg, extra={_RECORD_KEY: payload})

    def bind(self, **fields: Any) -> Logger:
        """Return a sub-logger with ``fields`` pre-attached to every line."""
        return Logger(self._log, {**self._bound, **fields})

    def with_component(self, name: str) -> Logger:
        """Return a sub-logger emitting ``component=name`` on every line."""
        return self.bind(component=name)


# --------------------------------------------------------------------------- #
# Module facade
# --------------------------------------------------------------------------- #


def configure(config: LoggerConfig | None = None, **overrides: Any) -> LoggerConfig:
    """Enable and configure SDK logging; return the active config.

    The SDK is **silent by default** — it emits nothing until logging is enabled
    via this function, :func:`set_level`, or the ``DISSEQT_LOG_LEVEL`` environment
    variable. Calling ``configure`` opts in and installs a redacting handler on
    the ``disseqt_ai_sdk`` root logger. Safe to call repeatedly (last call wins);
    only handlers this package owns are replaced, so an application's own handlers
    are never disturbed.

    Args:
        config: An explicit :class:`LoggerConfig`. When omitted, one is built
            from ``DISSEQT_*`` environment variables.
        **overrides: Field overrides applied on top of ``config`` / the env.

    Returns:
        The active :class:`LoggerConfig`.
    """
    global _config, _configured, _active
    with _lock:
        cfg = config if config is not None else LoggerConfig.from_env()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        _config = cfg
        _install_handlers(cfg, active=True)
        _configured = True
        _active = True
        return cfg


def disable() -> None:
    """Silence the SDK logger (opposite of :func:`configure` / :func:`set_level`).

    Replaces the output handler with a no-op sink so the SDK emits nothing. An
    application can re-enable later via :func:`configure` or :func:`set_level`.
    """
    global _configured, _active
    with _lock:
        _install_handlers(_config if _config is not None else LoggerConfig.from_env(), active=False)
        _configured = True
        _active = False


def _install_handlers(cfg: LoggerConfig, *, active: bool) -> None:
    """Install our root-logger handlers, replacing only the ones we own.

    Caller holds the lock. When ``active`` is False the SDK stays silent: only a
    ``NullHandler`` is attached and the level is raised past CRITICAL so a record
    is never even assembled. ``propagate`` is set False so SDK lines never leak
    into the application's root logger.
    """
    root = logging.getLogger(_ROOT_NAME)
    for handler in list(root.handlers):
        if getattr(handler, _OWNED_ATTR, False):
            root.removeHandler(handler)
    if active:
        formatter = _DisseqtFormatter(cfg)
        stream_handler: logging.Handler = logging.StreamHandler(cfg.stream)
        stream_handler.setFormatter(formatter)
        setattr(stream_handler, _OWNED_ATTR, True)
        root.addHandler(stream_handler)
        if cfg.file_path:
            file_handler = _make_file_handler(cfg, formatter)
            if file_handler is not None:
                root.addHandler(file_handler)
        root.setLevel(parse_level(cfg.level))
    else:
        null_handler = logging.NullHandler()
        setattr(null_handler, _OWNED_ATTR, True)
        root.addHandler(null_handler)
        root.setLevel(logging.CRITICAL + 1)  # silent: nothing passes the gate
    root.propagate = False


def _make_file_handler(cfg: LoggerConfig, formatter: logging.Formatter) -> logging.Handler | None:
    """Build a size-rotated file handler, or ``None`` on open failure."""
    import os
    from logging.handlers import RotatingFileHandler

    assert cfg.file_path is not None
    directory = os.path.dirname(cfg.file_path) or "."
    try:
        os.makedirs(directory, mode=0o750, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            cfg.file_path, maxBytes=cfg.file_max_bytes, backupCount=cfg.file_backups
        )
    except OSError as exc:
        print(
            f"[disseqt_logging] cannot open log file {cfg.file_path}: {exc}; stderr only",
            file=sys.stderr,
        )
        return None
    handler.setFormatter(formatter)
    setattr(handler, _OWNED_ATTR, True)
    return handler


def _ensure_configured() -> None:
    """Install handlers on first use. Silent unless ``DISSEQT_LOG_LEVEL`` is set.

    Honors the environment opt-in: when ``DISSEQT_LOG_LEVEL`` is present the SDK
    starts active; otherwise a silent ``NullHandler`` is attached and nothing is
    emitted until an app calls :func:`configure` / :func:`set_level`.
    """
    global _config, _configured, _active
    with _lock:
        if _configured:
            return
        cfg = LoggerConfig.from_env()
        active = bool(os.environ.get("DISSEQT_LOG_LEVEL", "").strip())
        _install_handlers(cfg, active=active)
        _config = cfg
        _active = active
        _configured = True


def get_logger(name: str | None = None) -> Logger:
    """Return a :class:`Logger` for ``name``, configuring the SDK on first use.

    Args:
        name: A module/component name (typically ``__name__``). Reparented under
            the ``disseqt_ai_sdk`` root so one handler serves the whole SDK; the
            prefix is stripped from the emitted ``logger`` field.

    Returns:
        A ready-to-use :class:`Logger`.
    """
    _ensure_configured()
    if not name or name == _ROOT_NAME:
        return Logger(logging.getLogger(_ROOT_NAME))
    child = name if name.startswith(_ROOT_NAME + ".") else f"{_ROOT_NAME}.{name}"
    return Logger(logging.getLogger(child))


def stdlib_logger(name: str | None = None) -> logging.Logger:
    """Return a standard-library :class:`logging.Logger` under the SDK root.

    Use this where a real ``logging.Logger`` is required (e.g. for backward
    compatibility) instead of the :class:`Logger` wrapper. Its records flow
    through the same shared handler and are rendered with the same schema and
    redaction; pass structured fields via ``extra={...}``. Every stdlib method
    (``setLevel``/``addHandler``/``isEnabledFor``/…) is available.

    Args:
        name: A module/component name. Reparented under the ``disseqt_ai_sdk``
            root so one handler serves the whole SDK.

    Returns:
        A configured stdlib :class:`logging.Logger`.
    """
    _ensure_configured()
    if not name or name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    child = name if name.startswith(_ROOT_NAME + ".") else f"{_ROOT_NAME}.{name}"
    return logging.getLogger(child)


def set_level(level: str | int) -> None:
    """Set the SDK log level, enabling output if it was silent.

    Calling this is an explicit opt-in: when the SDK is currently silent it swaps
    the no-op sink for a real redacting handler. Applies to every SDK logger.
    """
    global _config, _active
    with _lock:
        _ensure_configured()
        if not _active:
            cfg = _config if _config is not None else LoggerConfig.from_env()
            cfg.level = level
            _install_handlers(cfg, active=True)
            _config = cfg
            _active = True
        logging.getLogger(_ROOT_NAME).setLevel(parse_level(level))


def current_level() -> str:
    """Return the shared logger's current canonical level name."""
    _ensure_configured()
    return level_name(logging.getLogger(_ROOT_NAME).level)


def _default() -> Logger:
    """Return (and cache) the module-level default logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger()
    return _default_logger


def info(msg: str, **fields: Any) -> None:
    """Log at info level via the default logger."""
    _default().info(msg, **fields)


def warn(msg: str, **fields: Any) -> None:
    """Log at warning level via the default logger."""
    _default().warn(msg, **fields)


def error(msg: str, err: BaseException | None = None, **fields: Any) -> None:
    """Log at error level via the default logger."""
    _default().error(msg, err, **fields)


def debug(msg: str, **fields: Any) -> None:
    """Log at debug level via the default logger."""
    _default().debug(msg, **fields)
