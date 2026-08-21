"""
Auto-instrumentation entry points.

`instrument_all(client)` mirrors `langtrace.init(...)`: iterate every known
provider, skip anything that isn't installed, and patch what remains.
`instrument("openai", client)` targets a single provider.

Instrumentors are held per-client in a module-level registry so
`uninstrument*` can find them again to unpatch.
"""

from __future__ import annotations

import importlib
import threading
from typing import TYPE_CHECKING

from disseqt_agentic_sdk.instrumentation._registry import INSTRUMENTOR_CLASSES
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor
from disseqt_agentic_sdk.utils.logging import get_logger

if TYPE_CHECKING:
    from disseqt_agentic_sdk.client import DisseqtAgenticClient

logger = get_logger(__name__)

AVAILABLE_INSTRUMENTORS: list[str] = list(INSTRUMENTOR_CLASSES.keys())

# name → active instrumentor instance. One per provider (a second
# instrument() call on the same provider is a no-op unless uninstrumented
# first). Guarded by _LOCK so concurrent instrument_all() calls from
# multiple threads can't race the check-then-set.
_ACTIVE: dict[str, DisseqtInstrumentor] = {}
_LOCK = threading.RLock()


def instrument_all(client: DisseqtAgenticClient) -> list[str]:
    """
    Instrument every provider that's installed in the current environment.
    Returns the list of provider names that were successfully patched.
    """
    installed: list[str] = []
    for name in INSTRUMENTOR_CLASSES:
        if instrument(name, client):
            installed.append(name)
    logger.info(f"Auto-instrumented providers: {installed}")
    return installed


def instrument(name: str, client: DisseqtAgenticClient) -> bool:
    """
    Instrument a single provider by name. Silently returns False if the
    provider is unknown, its package isn't installed, or instrumentation
    fails.
    """
    dotted = INSTRUMENTOR_CLASSES.get(name)
    if dotted is None:
        logger.warning(f"unknown instrumentor: {name}")
        return False
    module_path, class_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        instrumentor_cls = getattr(module, class_name)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"could not load instrumentor {name}: {e}")
        return False
    instrumentor = instrumentor_cls()
    with _LOCK:
        if name in _ACTIVE:
            logger.debug(f"{name} already instrumented on this process")
            return False
        ok = instrumentor.instrument(client)
        if ok:
            _ACTIVE[name] = instrumentor
    return ok


def uninstrument(name: str) -> bool:
    """Remove patches for a single provider. Returns True if we did anything."""
    with _LOCK:
        instrumentor = _ACTIVE.pop(name, None)
    if instrumentor is None:
        return False
    instrumentor.uninstrument()
    return True


def uninstrument_all() -> None:
    """Remove patches for every provider we've instrumented."""
    with _LOCK:
        names = list(_ACTIVE.keys())
    for name in names:
        uninstrument(name)
