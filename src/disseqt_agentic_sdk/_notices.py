"""SDK-side notices surfaced at client construction time.

Follows the same channel contract as :mod:`disseqt_sdk._version` (v0.8.0):
plain stdlib logger under the package name, opt-out via env var read
once at import, fail-open so the notice channel can never break a
client. Customers silence via
``logging.getLogger("disseqt_agentic_sdk").setLevel(logging.ERROR)`` or
suppress a specific notice via its ``DISSEQT_SDK_DISABLE_*_NOTICE=1``
env var.
"""

from __future__ import annotations

import logging
import os
import threading

APPLICATIONS_REGISTRY_DOCS_URL = (
    "https://docs.disseqt.ai/docs/disseqt-sdk/agentic-observability/applications-registry"
)

_APPLICATION_ID_NOTICE_DISABLED = os.environ.get(
    "DISSEQT_SDK_DISABLE_APPLICATION_ID_NOTICE", ""
).strip().lower() in {"1", "true", "yes", "on"}

_notice_logger = logging.getLogger("disseqt_agentic_sdk")

_warned_missing_application_id = False
_warned_lock = threading.Lock()


def notify_missing_application_id() -> None:
    """Log the missing-application_id nudge at most once per process.

    Fail-open: the notice channel must never break a client, so any
    unexpected failure inside the log call is silently swallowed.
    """
    if _APPLICATION_ID_NOTICE_DISABLED:
        return
    global _warned_missing_application_id
    with _warned_lock:
        if _warned_missing_application_id:
            return
        _warned_missing_application_id = True
    try:
        _notice_logger.warning(
            "Please use application_id while sending spans. "
            "Check docs for more details: %s",
            APPLICATIONS_REGISTRY_DOCS_URL,
        )
    except Exception:  # noqa: BLE001 — notice channel is strictly best-effort
        pass


def _reset_for_tests() -> None:
    """Tests-only hook to clear the one-shot flag between test cases."""
    global _warned_missing_application_id
    with _warned_lock:
        _warned_missing_application_id = False
