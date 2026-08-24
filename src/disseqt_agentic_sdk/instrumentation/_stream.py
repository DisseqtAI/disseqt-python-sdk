"""
Stream wrappers.

Provider SDKs return generators (or async generators) for streaming
responses. We wrap them to (a) intercept each chunk so we can accumulate
the full text and token counts, and (b) close the span when the stream
finishes (or errors).

Two variants: `SyncStreamWrapper` for regular iterables, `AsyncStreamWrapper`
for async iterables. Providers pass in `on_chunk` to update state and
`on_finish` to write final attributes onto the span.

Error-handling policy: instrumentation callbacks (on_chunk / on_finish)
run under `contextlib.suppress(Exception)` — observability failures must
never propagate into the caller's stream loop.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from disseqt_agentic_sdk.instrumentation._utils import _SpanScope


class _StreamWrapperBase:
    """
    Shared state + finalize logic for the sync and async stream wrappers.

    The two concrete subclasses only differ in iteration protocol
    (``__iter__``/``__next__`` vs ``__aiter__``/``__anext__``) and in
    whether close forwarding awaits an ``aclose()`` coroutine. Everything
    else — constructor, sync ``_finish``, ``self._closed`` idempotency —
    lives here so a fix in one place is a fix in both. TP-2128 P4 #4.5.
    """

    def __init__(
        self,
        stream: Any,
        scope: _SpanScope,
        on_chunk: Callable[[Any], None],
        on_finish: Callable[[], None],
    ) -> None:
        self._stream = stream
        self._scope = scope
        self._on_chunk = on_chunk
        self._on_finish = on_finish
        self._closed = False

    def _run_on_finish(self) -> None:
        with contextlib.suppress(Exception):
            self._on_finish()

    def _finish(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Synchronous finalize path — used by ``SyncStreamWrapper`` and as a
        fallback on ``AsyncStreamWrapper`` for any direct sync caller.
        Forwards close() to the underlying stream so early exits don't
        leak the connection.
        """
        if self._closed:
            return
        self._closed = True
        self._run_on_finish()
        close = getattr(self._stream, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
        self._scope.__exit__(exc_type, exc_val, exc_tb)


class SyncStreamWrapper(_StreamWrapperBase):
    """Wraps a synchronous iterator; forwards chunks unchanged."""

    def __iter__(self) -> SyncStreamWrapper:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
        except StopIteration:
            self._finish(None, None, None)
            raise
        except BaseException as exc:
            # BaseException (not Exception) so `asyncio.CancelledError`,
            # `GeneratorExit`, and `KeyboardInterrupt` still finalize the
            # span. Client disconnect mid-stream is a normal production
            # event that raises CancelledError, not a bug.
            self._finish(type(exc), exc, exc.__traceback__)
            raise
        with contextlib.suppress(Exception):
            self._on_chunk(chunk)
        return chunk

    # Some SDKs support with-statement on their stream objects.
    def __enter__(self) -> SyncStreamWrapper:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._finish(exc_type, exc_val, exc_tb)

    def close(self) -> None:
        """Explicit close — forwards to the underlying stream's close()."""
        self._finish(None, None, None)


class AsyncStreamWrapper(_StreamWrapperBase):
    """Wraps an async iterator; forwards chunks unchanged."""

    def __aiter__(self) -> AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            await self._afinish(None, None, None)
            raise
        except BaseException as exc:
            # See sibling comment in SyncStreamWrapper.__next__.
            await self._afinish(type(exc), exc, exc.__traceback__)
            raise
        with contextlib.suppress(Exception):
            self._on_chunk(chunk)
        return chunk

    async def _afinish(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Async finalize path — prefers ``aclose()`` on async streams and
        falls back to ``close()`` for the rare stream that only exposes
        the sync variant. Guarded so a missing / buggy close never
        propagates into the caller.
        """
        if self._closed:
            return
        self._closed = True
        self._run_on_finish()
        aclose = getattr(self._stream, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await aclose()
        else:
            close = getattr(self._stream, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        self._scope.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> AsyncStreamWrapper:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self._afinish(exc_type, exc_val, exc_tb)

    async def aclose(self) -> None:
        """Explicit aclose — awaits underlying stream's aclose()."""
        await self._afinish(None, None, None)
