"""
User-supplied custom attributes on auto-instrumented spans.

When the SDK wraps `openai.chat.completions.create(...)`, it creates the
span internally — the user has no handle to attach their own context
(user_id, session_id, tenant, request_id, feature flag variant, etc.).
This module gives them one.

Usage — request-scoped:

    from disseqt_agentic_sdk.instrumentation import span_context

    with span_context(user_id="u_123", route="/checkout"):
        openai.chat.completions.create(...)
        anthropic.messages.create(...)
    # both spans emitted with `user_id` and `route` populated;
    # ambient bag is restored to its prior state on block exit.

Usage — long-lived (e.g. worker thread pins a tenant):

    from disseqt_agentic_sdk.instrumentation import (
        set_span_attributes, clear_span_attributes,
    )

    set_span_attributes(tenant="acme")
    # ... run tasks, every span they emit gets `tenant="acme"`
    clear_span_attributes()

Priority: user-supplied attributes are merged **at scope exit, after all
auto-emitted attributes have been written**. If a user sets a key that
collides with an auto key (e.g. ``agentic.request.model``), the user's
value overwrites — user intent always wins.

Async safety: backed by ``contextvars.ContextVar``. Two concurrent asyncio
tasks each see their own ambient bag; no bleed-through.

Threading — read this before assuming isolation:

* ``asyncio.loop.run_in_executor(...)`` copies the current context into
  the worker thread automatically, so ambient attributes set on the
  calling task are visible inside the executor task.
* **Bare** ``concurrent.futures.ThreadPoolExecutor().submit(fn, ...)``
  does **not** propagate context. Worker threads see an empty ambient
  bag, and any ``set_span_attributes(...)`` call inside the worker
  mutates the worker's context — never the caller's.
* For manual TPE use, wrap each submit with
  ``executor.submit(contextvars.copy_context().run, fn, ...)`` if you
  need the caller's ambient attributes inside the worker.

Recommendation: prefer the request-scoped ``span_context(...)`` helper
above ``set_span_attributes`` / ``clear_span_attributes``. It restores
prior state on exit and composes cleanly with async, sync, and executor
callers without you having to reason about which framework propagates
context.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# default=None avoids the mutable-default footgun; readers coalesce to {}.
# Every setter allocates a new dict so accidentally-shared references
# don't leak between contexts.
_ambient_attrs: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "disseqt_ambient_attrs", default=None
)


def _current() -> dict[str, Any]:
    """Read the ambient bag, coalescing None to an empty dict."""
    return _ambient_attrs.get() or {}


def set_span_attributes(**attributes: Any) -> None:
    """
    Merge ``attributes`` into the ambient bag for the current context.

    Later ``set_span_attributes`` calls override earlier ones for the same
    key. Every auto-instrumented span opened in this context will receive
    the full merged bag as span attributes at scope-exit time.

    Pass ``None`` as a value to opt out of writing that key (``safe_set``
    inside the merge skips None / empty strings).
    """
    if not attributes:
        return
    _ambient_attrs.set({**_current(), **attributes})


def clear_span_attributes() -> None:
    """Reset the ambient bag for the current context to empty."""
    _ambient_attrs.set({})


@contextmanager
def span_context(**attributes: Any) -> Iterator[None]:
    """
    Context manager: set ambient attributes on entry, restore on exit.

    Prefer this over the bare ``set_span_attributes`` when the scope is
    obvious (an HTTP request handler, a job execution) — it can't leak
    if the user forgets to clear.
    """
    token = _ambient_attrs.set({**_current(), **attributes})
    try:
        yield
    finally:
        _ambient_attrs.reset(token)


def _get_ambient_attrs() -> dict[str, Any]:
    """Internal: current ambient bag. Copy is not required; callers read only."""
    return _current()
