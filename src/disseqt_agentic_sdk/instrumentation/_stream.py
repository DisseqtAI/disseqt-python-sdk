"""
Stream wrappers.

Provider SDKs return generators (or async generators) for streaming
responses. We wrap them to (a) intercept each chunk so we can accumulate
the full text and token counts, and (b) close the span when the stream
finishes (or errors).

Two variants: `SyncStreamWrapper` for regular iterables, `AsyncStreamWrapper`
for async iterables. Providers pass in `on_chunk` to update state and
`on_finish` to write final attributes onto the span.
"""

from __future__ import annotations

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
        except Exception as exc:
            self._finish(type(exc), exc, exc.__traceback__)
            raise
        try:
            self._on_chunk(chunk)
        except Exception:  # noqa: BLE001 — never let observability break the caller
            pass
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
        try:
            self._on_finish()
        except Exception:  # noqa: BLE001
            pass
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
            self._finish(None, None, None)
            raise
        except Exception as exc:
            self._finish(type(exc), exc, exc.__traceback__)
            raise
        try:
            self._on_chunk(chunk)
        except Exception:  # noqa: BLE001
            pass
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
        try:
            self._on_finish()
        except Exception:  # noqa: BLE001
            pass
        self._scope.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> AsyncStreamWrapper:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._finish(exc_type, exc_val, exc_tb)
