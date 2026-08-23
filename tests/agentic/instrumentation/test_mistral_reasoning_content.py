"""
Regression tests for TP-2128 P1 #1.9.

Mistral reasoning-capable models (magistral-*, others) send
``delta.content`` and ``message.content`` as a **list of ContentChunk
objects** (TextChunk, ThinkChunk, ...) instead of a plain string. The
accumulator used to naively ``buffer.append(content)`` and then
``"".join(buffer)`` which crashes with `TypeError: sequence item ...:
expected str instance, list found` — swallowed inside
``contextlib.suppress(Exception)`` on the stream finalize path, so the
whole span lost model / response_id / tokens / finish_reason silently.

Tests exercise the ``_extract_content_text`` helper directly (fast,
provider-agnostic) plus the accumulator end-to-end.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from disseqt_agentic_sdk.instrumentation._oai_compat import (
    ChatStreamAccumulator,
    _extract_content_text,
)


class TestExtractContentText:
    def test_str_passthrough(self):
        assert _extract_content_text("hello") == "hello"

    def test_none(self):
        assert _extract_content_text(None) is None

    def test_list_of_text_chunks(self):
        # Real TextChunk shape: has `.text`.
        chunks = [
            SimpleNamespace(type="text", text="Hello "),
            SimpleNamespace(type="text", text="world."),
        ]
        assert _extract_content_text(chunks) == "Hello world."

    def test_list_with_think_chunk(self):
        # ThinkChunk.thinking is a list of Thinking(.text) blocks.
        chunks = [
            SimpleNamespace(
                type="thinking",
                thinking=[SimpleNamespace(text="hmm, let me consider ")],
            ),
            SimpleNamespace(type="text", text="the answer is 42."),
        ]
        assert _extract_content_text(chunks) == "hmm, let me consider the answer is 42."

    def test_unknown_chunk_type_falls_back_to_str(self):
        # Never lose visibility — record repr rather than dropping silently.
        chunks = [SimpleNamespace(type="custom", weird_field="opaque")]
        got = _extract_content_text(chunks)
        assert got is not None
        assert "opaque" in got or "custom" in got


class TestStreamAccumulatorWithListContent:
    def test_list_content_in_delta_does_not_crash_finalize(self):
        """
        Regression: buffer used to receive a list, `.join` crashed on
        finalize, contextlib.suppress swallowed it, no attrs landed.
        """
        acc = ChatStreamAccumulator()

        # Simulate two Mistral streaming chunks: first has list content
        # with a ThinkChunk, second has plain-string content.
        chunk_list_content = MagicMock(
            id="mistral-stream",
            model="magistral-latest",
            usage=None,
            choices=[
                MagicMock(
                    delta=MagicMock(
                        role="assistant",
                        content=[
                            SimpleNamespace(
                                type="thinking",
                                thinking=[SimpleNamespace(text="thinking... ")],
                            ),
                            SimpleNamespace(type="text", text="the answer is "),
                        ],
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ],
        )
        chunk_str_content = MagicMock(
            id="mistral-stream",
            model="magistral-latest",
            usage=None,
            choices=[
                MagicMock(
                    delta=MagicMock(role=None, content="42.", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )

        acc.absorb(chunk_list_content)
        acc.absorb(chunk_str_content)

        # Buffer must contain only strings — safe for "".join().
        assert all(
            isinstance(x, str) for x in acc.buffer
        ), f"buffer must be all-strings, got {[type(x).__name__ for x in acc.buffer]}"
        joined = "".join(acc.buffer)
        assert "thinking..." in joined
        assert "the answer is" in joined
        assert "42." in joined

    def test_finalize_populates_span_when_content_is_list(self):
        """
        End-to-end: before the fix, finalize crashed and the span had
        no output_messages / model / tokens. Now it lands cleanly.
        """
        span = MagicMock()
        acc = ChatStreamAccumulator()

        chunk = MagicMock(
            id="mistral-stream-2",
            model="magistral-latest",
            usage=MagicMock(prompt_tokens=3, completion_tokens=4, total_tokens=7),
            choices=[
                MagicMock(
                    delta=MagicMock(
                        role="assistant",
                        content=[SimpleNamespace(type="text", text="Paris.")],
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
        )
        acc.absorb(chunk)
        # Second chunk with usage only.
        usage_chunk = MagicMock(
            id="mistral-stream-2",
            model="magistral-latest",
            usage=MagicMock(prompt_tokens=3, completion_tokens=4, total_tokens=7),
            choices=[],
        )
        acc.absorb(usage_chunk)

        # Must not raise.
        acc.finalize(span)

        # set_token_usage was called with the parsed counts — proves
        # finalize actually reached that line (before the fix it crashed
        # on the join earlier).
        span.set_token_usage.assert_called_with(3, 4)
