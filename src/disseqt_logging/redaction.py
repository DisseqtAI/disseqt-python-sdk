"""PII / credential redaction for log fields.

A self-contained, standard-library-only port of the Disseqt platform's
``piiredact`` redaction rules. Two complementary strategies are applied:

* **Content-shape regexes** (:func:`redact_string`) — JWTs, emails,
  credit-card-shaped numbers, phone-shaped numbers, and long opaque tokens are
  replaced with a distinct placeholder so log/Sentry triage can tell which rule
  fired.
* **Field-name deny-list** (:func:`sensitive_key`) — a value stored under a key
  whose name contains a known-sensitive substring (``password``, ``token``,
  ``api_key``, ``authorization``, …) is replaced wholesale, catching short
  tokens and PHI that no regex would recognize.

The patterns mirror the platform rules character-for-character, including the
``re.ASCII`` compile flag (so ``\\b`` / ``\\d`` / ``\\w`` carry ASCII semantics
and a credential abutting a non-ASCII letter cannot escape the token rule).
"""

from __future__ import annotations

import re
from typing import Final

# Replacement tokens are public so log search can grep for them. Distinct
# placeholders let triage tell which strategy fired on which field.
EMAIL_TOKEN: Final = "[EMAIL]"
JWT_TOKEN: Final = "[JWT]"
CC_TOKEN: Final = "[CC]"
PHONE_TOKEN: Final = "[PHONE]"
TOKEN_LIKE_TOKEN: Final = "[TOKEN]"
SENSITIVE_VALUE_TOKEN: Final = "[REDACTED]"

# JWT: "eyJ" base64url header + "." + "eyJ" base64url payload + "." + signature.
_JWT_RE: Final = re.compile(r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+", re.ASCII)

# Email: RFC-conformant enough for redaction; over-match is the safe direction.
_EMAIL_RE: Final = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.ASCII)

# Credit-card-shaped: 13–19 digits, optionally split by single spaces/hyphens.
_CC_RE: Final = re.compile(r"\b\d(?:[ \-]?\d){12,18}\b", re.ASCII)

# Phone: optional leading +, 9+ digits with optional spaces/hyphens. Runs after
# CC so 13+ digit runs are already redacted as [CC].
_PHONE_RE: Final = re.compile(r"\+?\d(?:[ \-]?\d){8,}", re.ASCII)

# Long opaque token: 32+ chars of [A-Za-z0-9_-]. Catches API keys, session
# tokens, password hashes, and base64 payloads not matched by a finer shape.
_TOKEN_LIKE_RE: Final = re.compile(r"\b[A-Za-z0-9_\-]{32,}\b", re.ASCII)

# Case-insensitive substring deny-list checked against field names. Membership
# is substring, not exact, so "api_key", "apiKey", "API-Key", "reset_token",
# and "user_password_hash" all match. Entries are lowercase.
_SENSITIVE_KEY_SUBSTRINGS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "pwd",
    "token",
    "secret",
    "api_key",
    "apikey",
    "api-key",
    "private_key",
    "privatekey",
    "access_key",
    "access-key",
    "cookie",
    "authorization",
    "session",
    "credential",
    "signature",
    # Disseqt-platform-specific keys that have leaked in the past.
    "prompt",
    "csv",
    "upload",
    # SDK client identifiers (extends the platform deny-list): the validation
    # and agentic clients carry a project id that must never reach a log sink.
    # Both separator variants, matching the api_key / access_key convention.
    "project_id",
    "project-id",
)


def redact_string(value: str) -> str:
    """Redact content-shaped PII / credentials from ``value``.

    Patterns run in a fixed order: JWT and email first (they anchor on
    non-digit characters), then credit-card before phone (a 16-digit string
    should read ``[CC]`` not ``[PHONE]``), then the catch-all long-token rule.

    Args:
        value: The string to scrub.

    Returns:
        ``value`` with each recognized shape replaced by its placeholder.
    """
    value = _JWT_RE.sub(JWT_TOKEN, value)
    value = _EMAIL_RE.sub(EMAIL_TOKEN, value)
    value = _CC_RE.sub(CC_TOKEN, value)
    value = _PHONE_RE.sub(PHONE_TOKEN, value)
    value = _TOKEN_LIKE_RE.sub(TOKEN_LIKE_TOKEN, value)
    return value


def sensitive_key(name: str) -> bool:
    """Report whether ``name`` contains a sensitive substring (deny-list).

    Case-insensitive substring match. Use at log-emit sites with dynamic field
    names to decide whether a value must be replaced wholesale with
    :data:`SENSITIVE_VALUE_TOKEN`.

    Args:
        name: The field name to test.

    Returns:
        True if ``name`` (lower-cased) contains any denied substring.
    """
    lower = name.lower()
    return any(s in lower for s in _SENSITIVE_KEY_SUBSTRINGS)


# Nesting levels redact_field will descend into. Containers deeper than this
# are replaced wholesale (fail closed) rather than walked, which also bounds
# work on self-referential structures.
_MAX_REDACT_DEPTH: Final = 8


def redact_field(name: str, value: object) -> object:
    """Redact a structured-log field, recursing into containers.

    A string under a sensitive key becomes :data:`SENSITIVE_VALUE_TOKEN`; any
    other string is run through :func:`redact_string`. ``dict`` / ``list`` /
    ``tuple`` / ``set`` values are walked (up to a depth cap) so a secret
    nested inside a payload is scrubbed too: dict keys are checked with
    :func:`sensitive_key` at every level, and a sensitive parent key makes every
    string beneath it sensitive. Containers past the depth cap are replaced with
    :data:`SENSITIVE_VALUE_TOKEN`. Other non-string values (numbers, the logger's
    ``_Digest``, arbitrary objects) pass through untouched.

    Args:
        name: The field key.
        value: The field value.

    Returns:
        The redacted value (sets come back as lists).
    """
    return _redact(name, value, sensitive_key(name), 0)


def _redact(name: str, value: object, parent_sensitive: bool, depth: int) -> object:
    if isinstance(value, str):
        if parent_sensitive or sensitive_key(name):
            return SENSITIVE_VALUE_TOKEN
        return redact_string(value)
    if not isinstance(value, (dict, list, tuple, set, frozenset)):
        return value
    if depth >= _MAX_REDACT_DEPTH:
        return SENSITIVE_VALUE_TOKEN
    sensitive = parent_sensitive or sensitive_key(name)
    if isinstance(value, dict):
        return {
            k: _redact(k if isinstance(k, str) else str(k), v, sensitive, depth + 1)
            for k, v in value.items()
        }
    items = [_redact(name, v, sensitive, depth + 1) for v in value]
    if isinstance(value, tuple):
        return tuple(items)
    return items
