"""
Tests for the content-capture opt-out.

Some deployments (HIPAA / GDPR / any app whose tools might take
credentials as arguments) cannot ship prompt bodies, response text, or
tool-call arguments to the observability backend. This flag lets them
turn all of that off while keeping the non-content telemetry (model,
tokens, duration, tool names/ids, finish reasons) intact.

Regression coverage for TP-2128 P1 #1.2.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("openai")

from openai import OpenAI  # noqa: E402
from openai.types.chat import ChatCompletion, ChatCompletionMessage  # noqa: E402
from openai.types.chat.chat_completion import Choice  # noqa: E402
from openai.types.chat.chat_completion_message_tool_call import (  # noqa: E402
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage  # noqa: E402

from disseqt_agentic_sdk.instrumentation import (  # noqa: E402
    get_capture_content,
    instrument,
    set_capture_content,
    uninstrument,
)
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake_response_with_tools() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-x",
        model="gpt-4o-mini",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="here is my answer with SENSITIVE data",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(
                                name="send_email",
                                arguments='{"smtp_password":"hunter2"}',
                            ),
                        ),
                    ],
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


class TestCaptureContent:
    def test_default_is_capture_on(self, recording_client):
        # Default behavior — nothing changed. Content attrs present.
        assert get_capture_content() is True

        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            with patch.object(
                client.chat.completions,
                "_post",
                return_value=_fake_response_with_tools(),
                create=True,
            ):
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "can you send an email?"}],
                    tools=[
                        {
                            "type": "function",
                            "function": {"name": "send_email"},
                        }
                    ],
                )
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)

        # Every content attribute is present under default (opt-in) behavior.
        assert AgenticAttributes.INPUT_MESSAGES in attrs
        assert AgenticAttributes.OUTPUT_MESSAGES in attrs
        assert AgenticAttributes.TOOL_CALLS in attrs
        assert AgenticAttributes.TOOL_ARGS in attrs
        assert AgenticAttributes.REQUEST_TOOLS in attrs
        assert GenAIAttributes.PROMPT in attrs
        assert GenAIAttributes.COMPLETION in attrs
        assert GenAIAttributes.TOOL_CALLS in attrs
        # Sanity: sensitive data really was there when unopt-ed.
        assert "hunter2" in json.dumps(attrs)

    def test_disabled_strips_all_content_attrs(self, recording_client):
        """With capture off, no message body / tool args / tool schema attrs land."""
        original = get_capture_content()
        set_capture_content(False)
        try:
            instrument("openai", recording_client)
            try:
                client = OpenAI(api_key="fake")
                with patch.object(
                    client.chat.completions,
                    "_post",
                    return_value=_fake_response_with_tools(),
                    create=True,
                ):
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "can you send an email?"}],
                        tools=[
                            {
                                "type": "function",
                                "function": {"name": "send_email"},
                            }
                        ],
                    )
            finally:
                uninstrument("openai")
        finally:
            set_capture_content(original)

        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)

        # No content-bearing attributes wrote.
        for key in [
            AgenticAttributes.INPUT_MESSAGES,
            AgenticAttributes.OUTPUT_MESSAGES,
            AgenticAttributes.TOOL_CALLS,
            AgenticAttributes.TOOL_ARGS,
            AgenticAttributes.REQUEST_TOOLS,
            GenAIAttributes.PROMPT,
            GenAIAttributes.COMPLETION,
            GenAIAttributes.TOOL_CALLS,
            GenAIAttributes.TOOL_ARGS,
            GenAIAttributes.REQUEST_TOOLS,
        ]:
            assert key not in attrs, f"content attr {key!r} leaked when capture disabled"

        # Non-content telemetry is still captured.
        assert attrs[AgenticAttributes.REQUEST_MODEL] == "gpt-4o-mini"
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 5
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 5
        assert attrs[AgenticAttributes.USAGE_TOTAL_TOKENS] == 10
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "tool_calls"
        # Tool NAME / ID stay (metadata, not content) so validators keying
        # on identity still work.
        assert attrs[AgenticAttributes.TOOL_NAME] == "send_email"
        assert attrs[AgenticAttributes.TOOL_CALL_ID] == "call_1"

        # And crucially — the credential in tool arguments is nowhere in
        # the span payload.
        assert "hunter2" not in json.dumps(attrs)

    def test_concurrent_tasks_do_not_race_on_toggle(self):
        """
        TP-2128 round-2 P0 #0.1: `_capture_content` used to be a bare
        module-level bool. Two async tasks calling set_capture_content
        with opposite values would race — a secret-redacting task
        could have its ``safe_set`` land in the gap between another
        task's ``set_capture_content(False)`` and its own write,
        leaking the secret. Now it's a ContextVar so each task carries
        its own value.
        """
        import asyncio

        outer = get_capture_content()

        async def one_task(value: bool) -> bool:
            set_capture_content(value)
            # Yield twice to force interleaving. If capture were a
            # bare global, the sibling task's set would race in here
            # and change what we read back.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return get_capture_content()

        async def main() -> tuple[bool, bool]:
            return await asyncio.gather(one_task(False), one_task(True))

        try:
            a, b = asyncio.run(main())
            assert a is False, f"task A saw {a}, expected False — contextvar isolation broken"
            assert b is True, f"task B saw {b}, expected True — contextvar isolation broken"
            # Outer context untouched by inner set()s.
            assert get_capture_content() is outer
        finally:
            set_capture_content(outer)

    def test_plain_thread_inherits_startup_toggle(self):
        """
        TP-2128 round-3 senior review P0 #0.1: converting
        ``_capture_content`` to a bare ContextVar (round-2 P0 #0.1) fixed
        the concurrent-opposing-toggle race above, but a ContextVar's own
        value is only visible along the specific execution path that set
        it — a plain ``threading.Thread()`` spawned *after*
        ``set_capture_content(False)`` gets a brand-new, empty context and
        does NOT inherit it, silently defaulting back to capture-on. This
        is exactly the "flip the switch once at startup, then serve
        requests on a thread pool" pattern the SDK's own docstring
        recommends, and it's the shape most non-asyncio Python web
        servers use (threaded Flask, ThreadingHTTPServer, gunicorn
        --threads). The fix adds a process-wide fallback default that a
        context which never called set_capture_content() itself reads.
        """
        original = get_capture_content()
        set_capture_content(False)
        result: dict[str, bool] = {}
        try:

            def worker() -> None:
                # This thread never calls set_capture_content() itself —
                # it must still see the value set on the main thread
                # before it was spawned.
                result["seen"] = get_capture_content()

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=5)
        finally:
            set_capture_content(original)

        assert result.get("seen") is False, (
            "a plain thread spawned after set_capture_content(False) must still see "
            "capture disabled via the process-wide fallback default"
        )

    def test_plain_thread_end_to_end_span_is_redacted(self, recording_client):
        """
        Same scenario as test_plain_thread_inherits_startup_toggle, but
        through the real public manual-tracing API: a secret prompt
        traced via trace_llm_call() on a plain worker thread must not
        land on the span's attributes, even though the opt-out was only
        ever called on the main thread before the worker was spawned.
        (trace_llm_call applies the capture_content gate synchronously
        while building the span via set_messages_if_capturing — no need
        to close/flush the span to observe the redaction decision.)
        """
        from disseqt_agentic_sdk import start_trace
        from disseqt_agentic_sdk.api.helpers import trace_llm_call

        original = get_capture_content()
        set_capture_content(False)
        errors: list[BaseException] = []
        result: dict[str, Any] = {}

        def worker() -> None:
            try:
                with start_trace(recording_client, "worker_trace") as trace:
                    span = trace_llm_call(
                        trace,
                        name="chat",
                        model_name="gpt-4",
                        provider="openai",
                        input_messages=[{"role": "user", "content": "SSN_123-45-6789"}],
                    )
                    result["attributes"] = dict(span.attributes)
            except BaseException as exc:  # noqa: BLE001 - surface to the main thread
                errors.append(exc)

        try:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=5)
        finally:
            set_capture_content(original)

        assert not errors, f"worker thread raised: {errors}"
        attrs = result.get("attributes")
        assert attrs is not None, "worker thread never populated its span attributes"
        assert "SSN_123-45-6789" not in json.dumps(attrs, default=str), (
            "a plain worker thread must still honor a capture_content opt-out "
            "configured on the main thread before it was spawned"
        )

    def test_env_var_disables_at_import_time(self):
        """Setting DISSEQT_SDK_CAPTURE_CONTENT=0 before import → capture off."""
        import importlib

        from disseqt_agentic_sdk.instrumentation import _utils as utils_mod

        original_env = os.environ.get("DISSEQT_SDK_CAPTURE_CONTENT")
        original_flag = get_capture_content()
        os.environ["DISSEQT_SDK_CAPTURE_CONTENT"] = "0"
        try:
            importlib.reload(utils_mod)
            assert utils_mod.get_capture_content() is False
        finally:
            # Restore
            if original_env is None:
                os.environ.pop("DISSEQT_SDK_CAPTURE_CONTENT", None)
            else:
                os.environ["DISSEQT_SDK_CAPTURE_CONTENT"] = original_env
            importlib.reload(utils_mod)
            set_capture_content(original_flag)
