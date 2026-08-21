"""
Auto-instrumentation entry points.

`instrument_all(client)` mirrors `langtrace.init(...)`: iterate every known
provider, skip anything that isn't installed, and patch what remains.
`instrument("openai", client)` targets a single provider.

Instrumentors are held per-client in a module-level registry so
`uninstrument*` can find them again to unpatch.
"""

from __future__ import annotations

import contextlib
import importlib
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from disseqt_agentic_sdk.instrumentation._registry import INSTRUMENTOR_CLASSES
from disseqt_agentic_sdk.instrumentation.base import (
    REASON_ALREADY_INSTRUMENTED,
    REASON_INSTRUMENT_FAILURE,
    REASON_PACKAGE_MISSING,
    DisseqtInstrumentor,
    InstrumentationError,
)
from disseqt_agentic_sdk.utils.logging import get_logger

if TYPE_CHECKING:
    from disseqt_agentic_sdk.client import DisseqtAgenticClient

# Lifecycle-hook signatures. on_install fires after a successful patch,
# on_uninstall fires after a successful unpatch. Both are wrapped in
# contextlib.suppress so a bad user hook can't corrupt the registry.
OnInstallHook = Callable[[str, str], None]  # (provider_name, detected_version)
OnUninstallHook = Callable[[str], None]  # (provider_name,)

logger = get_logger(__name__)

AVAILABLE_INSTRUMENTORS: list[str] = list(INSTRUMENTOR_CLASSES.keys())

# name → active instrumentor instance. One per provider (a second
# instrument() call on the same provider is a no-op unless uninstrumented
# first). Guarded by _LOCK so concurrent instrument_all() calls from
# multiple threads can't race the check-then-set.
_ACTIVE: dict[str, DisseqtInstrumentor] = {}
_LOCK = threading.RLock()

# Reason codes that mean "nothing to do here" rather than "something is
# wrong". strict=True won't raise for these — they're the expected outcome
# when a provider SDK isn't installed or was already instrumented.
REASON_UNKNOWN_PROVIDER = "unknown_provider"
REASON_LOAD_FAILURE = "load_failure"
REASON_CLIENT_MISMATCH = "client_mismatch"
_SKIP_REASONS = frozenset({REASON_PACKAGE_MISSING, REASON_ALREADY_INSTRUMENTED})


def instrument_all(
    client: DisseqtAgenticClient,
    *,
    strict: bool = False,
    on_install: OnInstallHook | None = None,
) -> list[str]:
    """
    Instrument every provider that's installed in the current environment.
    Returns the list of provider names that were successfully patched.

    strict=True raises InstrumentationError on the first non-skip failure
    (unknown_provider, load_failure, unsupported_version, client_mismatch,
    instrument_failure). Package-not-installed and already-instrumented are
    still treated as skips because they're the expected mass-init outcome.

    on_install, if set, fires once per successfully patched provider with
    ``(provider_name, detected_version)``. Exceptions raised by the hook
    are swallowed — observability hooks must not break instrumentation.
    """
    installed: list[str] = []
    for name in INSTRUMENTOR_CLASSES:
        if instrument(name, client, strict=strict, on_install=on_install):
            installed.append(name)
    logger.info(f"Auto-instrumented providers: {installed}")
    return installed


def instrument(
    name: str,
    client: DisseqtAgenticClient,
    *,
    strict: bool = False,
    on_install: OnInstallHook | None = None,
) -> bool:
    """
    Instrument a single provider by name. Returns True on success, False on
    skip/failure.

    Called twice for the same provider:
    - Same client   → no-op, returns False, debug log.
    - Different client → warn loudly and refuse; the caller must call
      `uninstrument(name)` first if they intend to rebind.

    strict=True raises InstrumentationError instead of returning False for
    real failures (unknown_provider, load_failure, unsupported_version,
    client_mismatch, instrument_failure). Skips (package_missing,
    already_instrumented) still return False silently.

    on_install fires after successful patching with ``(name, version)``.
    """
    dotted = INSTRUMENTOR_CLASSES.get(name)
    if dotted is None:
        return _report(name, REASON_UNKNOWN_PROVIDER, "not in registry", strict)
    module_path, class_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        instrumentor_cls = getattr(module, class_name)
    except Exception as e:  # noqa: BLE001
        return _report(name, REASON_LOAD_FAILURE, str(e), strict)
    instrumentor = instrumentor_cls()
    with _LOCK:
        existing = _ACTIVE.get(name)
        if existing is not None:
            if existing._client is client:
                return _report(name, REASON_ALREADY_INSTRUMENTED, "same client", strict)
            return _report(
                name,
                REASON_CLIENT_MISMATCH,
                "different client bound; call uninstrument() first to rebind",
                strict,
            )
        ok = instrumentor.instrument(client)
        if ok:
            _ACTIVE[name] = instrumentor
            version = instrumentor.version
    if ok:
        if on_install is not None:
            with contextlib.suppress(Exception):
                on_install(name, version)
        return True
    reason, detail = instrumentor._last_error or (REASON_INSTRUMENT_FAILURE, "unknown")
    return _report(name, reason, detail, strict)


def _report(name: str, reason: str, detail: str, strict: bool) -> bool:
    """Log and, if strict, raise. Always returns False for the non-strict path."""
    if strict and reason not in _SKIP_REASONS:
        raise InstrumentationError(name, reason, detail)
    if reason in _SKIP_REASONS:
        logger.debug(f"{name}: {reason} ({detail})")
    else:
        logger.warning(f"{name}: {reason} ({detail})")
    return False


def uninstrument(name: str, *, on_uninstall: OnUninstallHook | None = None) -> bool:
    """
    Remove patches for a single provider. Returns True if we did anything.

    on_uninstall fires after successful unpatching with ``(name,)``.
    Exceptions from the hook are swallowed.
    """
    with _LOCK:
        instrumentor = _ACTIVE.pop(name, None)
    if instrumentor is None:
        return False
    instrumentor.uninstrument()
    if on_uninstall is not None:
        with contextlib.suppress(Exception):
            on_uninstall(name)
    return True


def uninstrument_all(*, on_uninstall: OnUninstallHook | None = None) -> None:
    """
    Remove patches for every provider we've instrumented.

    on_uninstall fires once per unpatched provider with ``(name,)``.
    """
    with _LOCK:
        names = list(_ACTIVE.keys())
    for name in names:
        uninstrument(name, on_uninstall=on_uninstall)


def get_instrumented_client(name: str) -> DisseqtAgenticClient | None:
    """Return the client currently bound to `name`, or None if not instrumented."""
    with _LOCK:
        instrumentor = _ACTIVE.get(name)
    return instrumentor._client if instrumentor is not None else None
