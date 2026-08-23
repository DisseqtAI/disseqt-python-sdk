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
