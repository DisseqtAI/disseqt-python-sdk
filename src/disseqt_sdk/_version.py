"""SDK version identity and the server-driven upgrade notice.

Single source of truth for the version: the installed ``disseqt-ai-sdk``
distribution metadata (i.e. whatever pip installed), so the reported
version can never drift from :file:`pyproject.toml`.

The upgrade notice deliberately logs through the plain stdlib
``disseqt_sdk`` logger rather than :mod:`disseqt_logging`: the shared
structured logger is silent until an application opts in, which would
bury a notice that must be visible by default (stdlib warnings reach
stderr via ``logging.lastResort`` even in unconfigured apps). The
``disseqt_sdk`` logger name is a documented contract — customers silence
the notice with ``logging.getLogger("disseqt_sdk").setLevel(logging.ERROR)``
or suppress it entirely with ``DISSEQT_SDK_DISABLE_VERSION_NOTICE=1``
(read once at import; the ``X-SDK-Version`` request header is still sent).
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version

try:
    SDK_VERSION = version("disseqt-ai-sdk")
except PackageNotFoundError:  # running from a source checkout
    SDK_VERSION = "0.0.0-dev"

USER_AGENT = f"disseqt-ai-sdk/{SDK_VERSION}"

_NOTICE_DISABLED = os.environ.get("DISSEQT_SDK_DISABLE_VERSION_NOTICE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_notice_logger = logging.getLogger("disseqt_sdk")

_warned_versions: set[str] = set()
_warned_lock = threading.Lock()


def sdk_identity_headers() -> dict[str, str]:
    """Return the request headers identifying this SDK build.

    ``X-SDK-Version`` is what production-monitoring's version middleware
    compares against the latest release; ``X-SDK-Lang`` names this SDK's
    release line so it is measured against the Python floor, never another
    language's (the middleware assumes Python when the header is absent,
    covering 0.8.0); ``User-Agent`` is the standard duplicate for
    gateway/access logs.
    """
    return {
        "X-SDK-Version": SDK_VERSION,
        "X-SDK-Lang": "python",
        "User-Agent": USER_AGENT,
    }


def check_version_notice(headers: Mapping[str, str]) -> None:
    """Warn (once per process per advertised version) when the server says
    a newer SDK exists.

    The server sets ``X-SDK-Latest-Version`` only when the caller's version
    is older than the latest release, so no client-side version comparison
    is needed; ``X-SDK-Notice``, when present (caller below the supported
    floor), is appended verbatim. Fail-open by construction: a version
    notice must never break or slow down an API call, so every failure
    path — absent or malformed headers included — is silence.
    """
    if _NOTICE_DISABLED:
        return
    try:
        latest = (headers.get("X-SDK-Latest-Version") or "").strip()
        if not latest:
            return
        with _warned_lock:
            if latest in _warned_versions:
                return
            _warned_versions.add(latest)
        message = (
            f"disseqt-ai-sdk {SDK_VERSION} is outdated; {latest} is available. "
            "Upgrade with: pip install -U disseqt-ai-sdk."
        )
        notice = (headers.get("X-SDK-Notice") or "").strip()
        if notice:
            message = f"{message} {notice}"
        _notice_logger.warning(message)
    except Exception:  # noqa: BLE001 — the notice channel is strictly best-effort
        pass
