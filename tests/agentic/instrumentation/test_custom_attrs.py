"""
Tests for the user-supplied custom-attributes API on auto-instrumented spans.

The contract:
- Attributes set via set_span_attributes / span_context land on every span
  the current context opens.
- User attributes are merged AFTER auto attributes at scope-exit time, so
  a user key that collides with an auto key overrides the auto value.
- Async tasks are isolated via contextvars — one task's ambient bag never
  bleeds into a concurrent task.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

pytest.importorskip("openai")

from openai import OpenAI  # noqa: E402
from openai.types.chat import ChatCompletion, ChatCompletionMessage  # noqa: E402
from openai.types.chat.chat_completion import Choice  # noqa: E402
from openai.types.completion_usage import CompletionUsage  # noqa: E402

from disseqt_agentic_sdk.instrumentation import (  # noqa: E402
    clear_span_attributes,
    instrument,
    set_span_attributes,
    span_context,
    uninstrument,
)
from disseqt_agentic_sdk.semantics import AgenticAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake() -> ChatCompletion:
    return ChatCompletion(
        id="c",
        model="gpt-4o-mini",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="ok"),
            )
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _call(client: OpenAI) -> None:
    with patch.object(client.chat.completions, "_post", return_value=_fake(), create=True):
        client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}]
        )


class TestCustomAttrs:
    def test_ambient_attrs_land_on_auto_span(self, recording_client):
        instrument("openai", recording_client)
        try:
            set_span_attributes(user_id="u_123", tenant="acme")
            try:
                _call(OpenAI(api_key="fake"))
            finally:
                clear_span_attributes()
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)
        assert attrs["user_id"] == "u_123"
        assert attrs["tenant"] == "acme"

    def test_user_attr_overrides_auto_attr(self, recording_client):
        # User sets a key that our auto instrumentation also sets; user wins.
        instrument("openai", recording_client)
        try:
            set_span_attributes(**{AgenticAttributes.REQUEST_MODEL: "override-model"})
            try:
                _call(OpenAI(api_key="fake"))
            finally:
                clear_span_attributes()
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)
        # Auto path sets REQUEST_MODEL="gpt-4o-mini". User override lands
        # in the __exit__ merge and overwrites.
        assert attrs[AgenticAttributes.REQUEST_MODEL] == "override-model"

    def test_span_context_scopes_and_restores(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            # Outside the block: no ambient attrs.
            _call(client)
            with span_context(request_id="req-1"):
                _call(client)
            # After the block: ambient restored.
            _call(client)
        finally:
            uninstrument("openai")

        spans = [
            s
            for s in recording_client.buffer.spans  # type: ignore[attr-defined]
            if s.name == "openai.chat.completions.create"
        ]
        assert len(spans) == 3
        outer_before = json.loads(spans[0].attributes_json)
        inside = json.loads(spans[1].attributes_json)
        outer_after = json.loads(spans[2].attributes_json)
        assert "request_id" not in outer_before
        assert inside["request_id"] == "req-1"
        assert "request_id" not in outer_after

    def test_bare_threadpool_does_not_propagate_ambient_attrs(self, recording_client):
        """
        TP-2128 P2 #2.1: worker threads see an empty ambient bag when the
        caller uses bare ThreadPoolExecutor.submit. This is a Python
        contextvars fact, not a bug in the SDK — but the docstring used
        to imply TPE isolation "just worked", so this test pins the real
        behavior and the mitigation.
        """
        import contextvars
        from concurrent.futures import ThreadPoolExecutor

        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")

            def _worker(tag: str) -> None:
                # Whatever the caller set is invisible unless we propagate.
                with span_context(task_tag=tag):
                    _call(client)

            set_span_attributes(outer="from-main-thread")
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    # Bare submit: outer="from-main-thread" is NOT copied.
                    ex.submit(_worker, "bare").result()
                    # copy_context().run: outer IS copied.
                    ctx = contextvars.copy_context()
                    ex.submit(ctx.run, _worker, "copied").result()
            finally:
                clear_span_attributes()
        finally:
            uninstrument("openai")

        spans = [
            s
            for s in recording_client.buffer.spans  # type: ignore[attr-defined]
            if s.name == "openai.chat.completions.create"
        ]
        by_tag = {
            json.loads(s.attributes_json)["task_tag"]: json.loads(s.attributes_json) for s in spans
        }
        # Bare submit: no outer key visible in the worker.
        assert "outer" not in by_tag["bare"], (
            "bare TPE.submit must NOT propagate main-thread ambient attrs — "
            "if this ever starts propagating, update the _custom_attrs.py docstring"
        )
        # Wrapped submit: outer copied into the worker context.
        assert by_tag["copied"].get("outer") == "from-main-thread"

    def test_docstring_documents_tpe_caveat(self):
        """Docstring must warn about bare TPE. Guards against future regressions."""
        from disseqt_agentic_sdk.instrumentation import _custom_attrs

        doc = _custom_attrs.__doc__ or ""
        assert "ThreadPoolExecutor" in doc
        assert "copy_context" in doc
        assert "span_context" in doc

    def test_contextvars_isolate_concurrent_async_tasks(self, recording_client):
        # Two asyncio tasks set different ambient attrs. Neither should
        # see the other's — contextvars propagate per-task copies.
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")

            async def one_call(tag: str) -> None:
                with span_context(task_tag=tag):
                    await asyncio.sleep(0)  # yield to force interleaving
                    _call(client)
                    await asyncio.sleep(0)

            async def main() -> None:
                await asyncio.gather(one_call("alpha"), one_call("beta"))

            asyncio.run(main())
        finally:
            uninstrument("openai")

        spans = [
            s
            for s in recording_client.buffer.spans  # type: ignore[attr-defined]
            if s.name == "openai.chat.completions.create"
        ]
        assert len(spans) == 2
        tags = sorted(json.loads(s.attributes_json)["task_tag"] for s in spans)
        assert tags == ["alpha", "beta"]
