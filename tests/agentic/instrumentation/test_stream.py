"""
Direct tests for SyncStreamWrapper / AsyncStreamWrapper.

These exercise the wrappers in isolation from any provider, focused on
the exception-handling contract: any exception raised by the underlying
iterator — including ``asyncio.CancelledError`` (BaseException, not
Exception) — must trigger ``on_finish`` so the span is finalized before
the exception propagates.

Regression coverage for TP-2128 P0 blocker #0.2: `except Exception`
missed CancelledError, so client-disconnect mid-stream leaked spans.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from disseqt_agentic_sdk.instrumentation._stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)


class _RecordingScope:
    """Minimal _SpanScope stand-in; records what __exit__ was called with."""

    def __init__(self) -> None:
        self.span = MagicMock()
        self.exit_calls: list[tuple[Any, Any, Any]] = []

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
        self.exit_calls.append((exc_type, exc_val, exc_tb))


class TestSyncStreamCancellation:
    def test_base_exception_from_iterator_triggers_finish(self):
        """KeyboardInterrupt is a BaseException. It must still finalize."""
        scope = _RecordingScope()
        finish_calls: list[bool] = []

        def _boom():
            yield 1
            raise KeyboardInterrupt("simulated user Ctrl-C")

        wrapper = SyncStreamWrapper(
            stream=_boom(),
            scope=scope,  # type: ignore[arg-type]
            on_chunk=lambda _chunk: None,
            on_finish=lambda: finish_calls.append(True),
        )

        assert next(wrapper) == 1
        try:
            next(wrapper)
        except KeyboardInterrupt:
            pass
        assert finish_calls == [True], "on_finish must fire even for BaseException"
        assert len(scope.exit_calls) == 1
        assert scope.exit_calls[0][0] is KeyboardInterrupt


class TestSyncStreamClose:
    """
    TP-2128 P2 #2.6: SyncStreamWrapper must forward close() to the
    underlying provider stream. Early exits (with-statement break, caller
    exception before StopIteration) otherwise leak the HTTP connection.
    """

    def test_context_manager_exit_closes_underlying_stream(self):
        scope = _RecordingScope()
        closed: list[bool] = []

        class _Stream:
            def __init__(self):
                self._i = iter([1, 2, 3, 4, 5])

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._i)

            def close(self):
                closed.append(True)

        with SyncStreamWrapper(
            stream=_Stream(),
            scope=scope,  # type: ignore[arg-type]
            on_chunk=lambda _c: None,
            on_finish=lambda: None,
        ) as wrapper:
            # Consume 2 chunks then break — early exit path.
            next(wrapper)
            next(wrapper)
        assert closed == [True], "underlying stream.close() must run on wrapper exit"

    def test_close_method_forwards(self):
        scope = _RecordingScope()
        closed: list[bool] = []

        class _Stream:
            def __iter__(self):
                return self

            def __next__(self):
                return 42

            def close(self):
                closed.append(True)

        wrapper = SyncStreamWrapper(
            stream=_Stream(),
            scope=scope,  # type: ignore[arg-type]
            on_chunk=lambda _c: None,
            on_finish=lambda: None,
        )
        wrapper.close()
        assert closed == [True]

    def test_stream_without_close_does_not_crash(self):
        scope = _RecordingScope()

        wrapper = SyncStreamWrapper(
            stream=iter([1, 2]),  # bare generator: no close()
            scope=scope,  # type: ignore[arg-type]
            on_chunk=lambda _c: None,
            on_finish=lambda: None,
        )
        wrapper.close()  # must not raise


class TestAsyncStreamClose:
    """TP-2128 P2 #2.6: AsyncStreamWrapper must forward aclose()."""

    def test_aexit_awaits_aclose(self):
        scope = _RecordingScope()
        aclosed: list[bool] = []

        class _AStream:
            def __init__(self):
                self._i = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._i >= 5:
                    raise StopAsyncIteration
                self._i += 1
                return self._i

            async def aclose(self):
                aclosed.append(True)

        wrapper = AsyncStreamWrapper(
            stream=_AStream(),
            scope=scope,  # type: ignore[arg-type]
            on_chunk=lambda _c: None,
            on_finish=lambda: None,
        )

        async def _drive():
            async with wrapper as w:
                await w.__anext__()
                # break early — aclose must still run on __aexit__.

        asyncio.run(_drive())
        assert aclosed == [True], "underlying stream.aclose() must run on __aexit__"


class TestAsyncStreamCancellation:
    def test_cancellation_via_wait_for_finalizes_span(self):
        """
        Slow async iterator + asyncio.wait_for(timeout) raises
        CancelledError inside __anext__. Before the fix, the `except
        Exception` branch didn't catch it and the span leaked. Now
        `except BaseException` finalizes.
        """
        scope = _RecordingScope()
        finish_calls: list[bool] = []

        async def _slow():
            # First chunk is instant so we get past __aiter__ setup, then
            # hang so wait_for cancels us.
            yield 0
            await asyncio.sleep(10)
            yield 1  # never reached

        wrapper = AsyncStreamWrapper(
            stream=_slow(),
            scope=scope,  # type: ignore[arg-type]
            on_chunk=lambda _chunk: None,
            on_finish=lambda: finish_calls.append(True),
        )

        async def _drive():
            # Read one chunk to prime the pump.
            first = await wrapper.__anext__()
            assert first == 0
            # Now wait_for the next chunk with an impossibly-short timeout
            # so asyncio cancels the pending __anext__.
            try:
                await asyncio.wait_for(wrapper.__anext__(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

        asyncio.run(_drive())

        assert finish_calls == [True], "on_finish must fire when cancelled"
        assert len(scope.exit_calls) == 1
        # The exception recorded should be CancelledError (or its wrapper),
        # not None — proves _finish saw the cancel not a clean end.
        recorded_type = scope.exit_calls[0][0]
        assert recorded_type is not None and issubclass(recorded_type, BaseException)
