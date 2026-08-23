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


class SyncStreamWrapper:
    """Wraps a synchronous iterator; forwards chunks unchanged."""

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

    def _finish(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._on_finish()
        # Forward close() to the underlying provider stream. Early exits
        # (with-statement break, exception in the caller's loop) otherwise
        # leave the HTTP connection open. Not every stream object exposes
        # close(); guard both the getattr and the call.
        close = getattr(self._stream, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
        self._scope.__exit__(exc_type, exc_val, exc_tb)

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


class AsyncStreamWrapper:
    """Wraps an async iterator; forwards chunks unchanged."""

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

    def __aiter__(self) -> AsyncStreamWrapper:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            await self._afinish(None, None, None)
            raise
        except BaseException as exc:
            # BaseException (not Exception) so `asyncio.CancelledError`,
            # `GeneratorExit`, and `KeyboardInterrupt` still finalize the
            # span. Client disconnect mid-stream is a normal production
            # event that raises CancelledError, not a bug.
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
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._on_finish()
        # Prefer aclose() on async streams; fall back to close() for the
        # (rare) case where the provider stream exposes only sync close.
        # Guard both the getattr and the call so a missing method or a
        # buggy close never propagates into the caller.
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

    def _finish(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Kept for compatibility with any direct sync callers. Prefer
        # _afinish so the async close() runs.
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._on_finish()
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
