"""Local credential store — ``~/.disseqt/config.json``.

Read/write helpers with strict POSIX perms:
- config file: ``0600``
- parent dir:  ``0700``

Refuses to :func:`load` a file whose mode is wider than ``0600``, so a
world-readable stash of credentials fails loudly instead of silently
seeding requests. Shape::

    {
      "auth": {
        "api_key":    "sk_live_...",
        "project_id": "...",
        "base_url":   "https://api.disseqt.ai/realtime-validations"
      }
    }
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".disseqt"
CONFIG_PATH = CONFIG_DIR / "config.json"

# 0o600 file, 0o700 dir — owner-only rw / rwx.
_FILE_MODE = 0o600
_DIR_MODE = 0o700
_PERM_MASK = 0o777


class AuthConfigPermissionError(Exception):
    """Raised when the on-disk config is readable by anyone but the owner."""


class AuthMissingError(Exception):
    """Raised when a :class:`Client` is constructed with no resolvable creds.

    Fails at construction rather than deferring to the first API call, so
    misconfiguration surfaces where it is caused, not where it is used.
    """


def load() -> dict[str, Any] | None:
    """Return the stored auth dict, or ``None`` if no config exists.

    Raises:
        AuthConfigPermissionError: The config file's POSIX mode is wider
            than ``0600``. Never returns creds from an insecure file.
        ValueError: The file exists but is not valid JSON.
    """
    if not CONFIG_PATH.exists():
        return None
    _ensure_owner_only(CONFIG_PATH)
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"config at {CONFIG_PATH} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        return None
    auth = data.get("auth")
    return auth if isinstance(auth, dict) else None


def save(auth: dict[str, Any]) -> None:
    """Write ``auth`` to the config file with ``0600`` perms.

    Creates the parent dir (``0700``) if missing. Overwrites atomically.
    """
    CONFIG_DIR.mkdir(mode=_DIR_MODE, exist_ok=True)
    # Tighten in case the dir already existed with wider perms.
    try:
        os.chmod(CONFIG_DIR, _DIR_MODE)
    except OSError:
        pass  # Non-POSIX filesystem; carry on.

    payload = json.dumps({"auth": auth}, indent=2, sort_keys=True)
    # Write via a temp file + rename so we never leave a half-written config.
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, CONFIG_PATH)
    try:
        os.chmod(CONFIG_PATH, _FILE_MODE)
    except OSError:
        pass


def clear() -> None:
    """Delete the config file. Idempotent — missing file is a no-op."""
    CONFIG_PATH.unlink(missing_ok=True)


def _ensure_owner_only(path: Path) -> None:
    """Raise :class:`AuthConfigPermissionError` if perms are wider than 0600.

    Silent no-op on non-POSIX platforms (Windows) where ``st_mode`` bits
    don't map cleanly.
    """
    if os.name != "posix":
        return
    mode = stat.S_IMODE(path.stat().st_mode) & _PERM_MASK
    if mode & 0o077:
        raise AuthConfigPermissionError(
            f"{path} has permissions {oct(mode)}; expected 0o600. " f"Run: chmod 600 {path}"
        )
