"""
Unit tests for helper functions.
"""

from unittest.mock import patch

import pytest

from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
from disseqt_agentic_sdk.api.client import get_client, set_client
from disseqt_agentic_sdk.api.helpers import (
    trace_agent_action,
    trace_function,
    trace_llm_call,
    trace_tool_call,
)
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.semantics import AgenticAttributes, AgenticOperation


class TestHelpers:
    """Tests for helper functions."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        with (
            patch("disseqt_agentic_sdk.client.client.HTTPTransport"),
            patch("disseqt_agentic_sdk.client.client.TraceBuffer"),
        ):
            self.client = DisseqtAgenticClient(
                api_key="test_key",
                project_id="test_proj",
                service_name="test_service",
                application_id="test-app-id",
            )
            yield
            try:
                self.client.shutdown()
            except Exception:
                pass

    def test_trace_llm_call_basic(self):
        """Test basic LLM call tracing."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_llm_call(
                trace, name="chat_completion", model_name="gpt-4", provider="openai"
            )

            assert span.name == "chat_completion"
            assert span.kind == SpanKind.MODEL_EXEC.value
            assert span.attributes[AgenticAttributes.REQUEST_MODEL] == "gpt-4"
            assert span.attributes[AgenticAttributes.PROVIDER_NAME] == "openai"
            assert span.attributes[AgenticAttributes.OPERATION_NAME] == AgenticOperation.CHAT

    def test_trace_llm_call_with_messages(self):
        """Test LLM call tracing with messages."""
        input_msgs = [{"role": "user", "content": "Hello"}]
        output_msgs = [{"role": "assistant", "content": "Hi"}]

        with start_trace(self.client, "test_trace") as trace:
            span = trace_llm_call(
                trace,
                name="chat",
                model_name="gpt-4",
                provider="openai",
                input_messages=input_msgs,
                output_messages=output_msgs,
            )

            assert span.attributes[AgenticAttributes.INPUT_MESSAGES] == input_msgs
            assert span.attributes[AgenticAttributes.OUTPUT_MESSAGES] == output_msgs

    def test_trace_llm_call_with_tokens(self):
        """Test LLM call tracing with token usage."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_llm_call(
                trace,
                name="chat",
                model_name="gpt-4",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
            )

            assert span.attributes[AgenticAttributes.USAGE_INPUT_TOKENS] == 100
            assert span.attributes[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 50

    def test_trace_llm_call_with_parameters(self):
        """Test LLM call tracing with temperature and max_tokens."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_llm_call(
                trace,
                name="chat",
                model_name="gpt-4",
                provider="openai",
                temperature=0.7,
                max_tokens=200,
            )

            assert span.attributes[AgenticAttributes.REQUEST_TEMPERATURE] == 0.7
            assert span.attributes[AgenticAttributes.REQUEST_MAX_TOKENS] == 200

    def test_trace_llm_call_with_kwargs(self):
        """Test LLM call tracing with additional kwargs."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_llm_call(
                trace,
                name="chat",
                model_name="gpt-4",
                provider="openai",
                custom_attr="custom_value",
                another_attr=123,
            )

            assert span.attributes["custom_attr"] == "custom_value"
            assert span.attributes["another_attr"] == 123

    def test_trace_agent_action_basic(self):
        """Test basic agent action tracing."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_agent_action(trace, name="planning", agent_name="assistant")

            assert span.name == "planning"
            assert span.kind == SpanKind.AGENT_EXEC.value
            assert span.attributes[AgenticAttributes.AGENT_NAME] == "assistant"

    def test_trace_agent_action_with_id_and_version(self):
        """Test agent action tracing with ID and version."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_agent_action(
                trace,
                name="planning",
                agent_name="assistant",
                agent_id="agent_001",
                agent_version="1.0.0",
            )

            assert span.attributes[AgenticAttributes.AGENT_NAME] == "assistant"
            assert span.attributes[AgenticAttributes.AGENT_ID] == "agent_001"
            assert span.attributes[AgenticAttributes.AGENT_VERSION] == "1.0.0"

    def test_trace_agent_action_with_operation(self):
        """Test agent action tracing with operation."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_agent_action(
                trace,
                name="planning",
                agent_name="assistant",
                operation=AgenticOperation.INVOKE_AGENT,
            )

            assert (
                span.attributes[AgenticAttributes.OPERATION_NAME] == AgenticOperation.INVOKE_AGENT
            )

    def test_trace_agent_action_with_kwargs(self):
        """Test agent action tracing with additional kwargs."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_agent_action(
                trace, name="planning", agent_name="assistant", custom_key="custom_value"
            )

            assert span.attributes["custom_key"] == "custom_value"

    def test_trace_tool_call_basic(self):
        """Test basic tool call tracing."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_tool_call(trace, name="weather_api", tool_name="get_weather")

            assert span.name == "weather_api"
            assert span.kind == SpanKind.TOOL_EXEC.value
            assert span.attributes[AgenticAttributes.TOOL_NAME] == "get_weather"
            assert (
                span.attributes[AgenticAttributes.OPERATION_NAME] == AgenticOperation.EXECUTE_TOOL
            )

    def test_trace_tool_call_with_call_id(self):
        """Test tool call tracing with call ID."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_tool_call(
                trace, name="weather_api", tool_name="get_weather", call_id="call_001"
            )

            assert span.attributes[AgenticAttributes.TOOL_CALL_ID] == "call_001"

    def test_trace_tool_call_with_definitions(self):
        """Test tool call tracing with tool definitions."""
        tool_defs = [{"name": "get_weather", "description": "Get weather"}]

        with start_trace(self.client, "test_trace") as trace:
            span = trace_tool_call(
                trace, name="weather_api", tool_name="get_weather", tool_definitions=tool_defs
            )

            assert span.attributes[AgenticAttributes.TOOL_DEFINITIONS] == tool_defs

    def test_trace_tool_call_with_kwargs(self):
        """Test tool call tracing with additional kwargs."""
        with start_trace(self.client, "test_trace") as trace:
            span = trace_tool_call(
                trace, name="weather_api", tool_name="get_weather", custom_attr="value"
            )

            assert span.attributes["custom_attr"] == "value"

    def test_manual_helpers_honor_capture_content_off(self):
        """
        TP-2128 round-2 P0 #0.2 + P1 #1.1: the four public manual-
        tracing helpers used to write straight to span.set_messages /
        span.set_attribute, bypassing the capture_content gate that
        auto-instrumentation honored. A deployment calling
        set_capture_content(False) for compliance got zero redaction
        on any manual-trace path — the exact opposite of the feature's
        contract. Fix: route trace_llm_call through
        set_messages_if_capturing, and the *_action / *_tool_call /
        *_function paths through safe_set. Also add TOOL_DEFINITIONS
        to _CONTENT_ATTR_KEYS so tool-schema credentials are covered.
        """
        import json as _json

        from disseqt_agentic_sdk.instrumentation import (
            get_capture_content,
            set_capture_content,
        )

        original = get_capture_content()
        set_capture_content(False)
        try:
            with start_trace(self.client, "test_trace") as trace:
                # 1. trace_llm_call — messages must not land on the span.
                llm_span = trace_llm_call(
                    trace,
                    name="chat",
                    model_name="gpt-4",
                    provider="openai",
                    input_messages=[{"role": "user", "content": "MY_SECRET_PROMPT"}],
                    output_messages=[{"role": "assistant", "content": "MY_SECRET_COMPLETION"}],
                )
                llm_dump = _json.dumps(llm_span.attributes, default=str)
                assert (
                    "MY_SECRET_PROMPT" not in llm_dump
                ), "trace_llm_call must route input_messages through the gate"
                assert (
                    "MY_SECRET_COMPLETION" not in llm_dump
                ), "trace_llm_call must route output_messages through the gate"
                # Non-content attrs (model, provider) must still land.
                assert llm_span.attributes[AgenticAttributes.REQUEST_MODEL] == "gpt-4"

                # 2. trace_tool_call — tool_definitions with a fake
                # credential in a parameter default must not land.
                tool_span = trace_tool_call(
                    trace,
                    name="email_api",
                    tool_name="send_email",
                    tool_definitions=[
                        {
                            "name": "send_email",
                            "parameters": {"smtp_password": "hunter2"},
                        }
                    ],
                )
                tool_dump = _json.dumps(tool_span.attributes, default=str)
                assert "hunter2" not in tool_dump, (
                    "trace_tool_call must gate tool_definitions "
                    "(P1 #1.1 adds TOOL_DEFINITIONS to _CONTENT_ATTR_KEYS)"
                )
                # Non-content attrs land.
                assert tool_span.attributes[AgenticAttributes.TOOL_NAME] == "send_email"

                # 3. trace_agent_action — a content-shaped kwarg
                # (INPUT_MESSAGES) must not land.
                agent_span = trace_agent_action(
                    trace,
                    name="plan",
                    agent_name="assistant",
                    **{AgenticAttributes.INPUT_MESSAGES: "MY_SECRET_AGENT_MSG"},
                )
                agent_dump = _json.dumps(agent_span.attributes, default=str)
                assert "MY_SECRET_AGENT_MSG" not in agent_dump

                # Non-content kwargs land (agent_name is set via
                # set_agent_info, not the kwargs loop).
                assert agent_span.attributes[AgenticAttributes.AGENT_NAME] == "assistant"

            # 4. trace_function decorator — content-shaped span_attr
            # must not land either. TP-2128 round-3 senior review P3
            # #3.1: this block used to just call the decorated function
            # without asserting anything, so a revert of trace_function's
            # own safe_set() fix (back to a raw span.set_attribute())
            # would have gone undetected. Swap in a RecordingBuffer so
            # the span this decorator closes is actually introspectable.
            from tests.agentic.instrumentation.conftest import (
                RecordingBuffer,
                find_span,
            )

            recording_buffer = RecordingBuffer()
            original_buffer = self.client.buffer
            self.client.buffer = recording_buffer
            try:

                @trace_function(
                    self.client,
                    name="dec_fn",
                    **{AgenticAttributes.INPUT_MESSAGES: "MY_SECRET_FN_MSG"},
                )
                def _fn():
                    return 1

                _fn()

                fn_span = find_span(self.client, "dec_fn")
                # attributes_json is already a serialized JSON string —
                # a plain substring check is enough here.
                assert (
                    "MY_SECRET_FN_MSG" not in fn_span.attributes_json
                ), "trace_function must route content-shaped span_attrs through the gate"
            finally:
                self.client.buffer = original_buffer
        finally:
            set_capture_content(original)

    def test_trace_function_decorator_basic(self):
        """Test trace_function decorator."""

        @trace_function(self.client, name="my_function")
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"

    def test_trace_function_decorator_with_kind(self):
        """Test trace_function decorator with custom kind."""

        @trace_function(self.client, name="agent_func", kind=SpanKind.AGENT_EXEC)
        def agent_function():
            return "agent_result"

        result = agent_function()
        assert result == "agent_result"

    def test_trace_function_decorator_with_attrs(self):
        """Test trace_function decorator with attributes."""

        @trace_function(self.client, name="func", agent_name="test_agent")
        def test_func():
            return "test"

        result = test_func()
        assert result == "test"

    def test_trace_function_decorator_with_exception(self):
        """Test trace_function decorator handles exceptions."""

        @trace_function(self.client, name="error_func")
        def error_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            error_function()

    def test_trace_function_decorator_without_init(self):
        """Test trace_function decorator - client is always required now."""

        # Client is required, so this test just verifies it works
        @trace_function(self.client, name="func")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

    def test_trace_function_decorator_default_name(self):
        """Test trace_function decorator uses function name when name not provided."""

        @trace_function(self.client)
        def my_custom_function():
            return "result"

        result = my_custom_function()
        assert result == "result"

    def test_trace_function_decorator_string_kind(self):
        """Test trace_function decorator with string kind."""

        @trace_function(self.client, name="func", kind="MODEL_EXEC")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

    def test_trace_function_decorator_custom_span_kind(self):
        """Test trace_function decorator with custom span kind string."""

        @trace_function(self.client, name="func", kind="CUSTOM_OPERATION")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

    def test_trace_function_decorator_invalid_enum_keeps_string(self):
        """Test trace_function decorator keeps invalid enum as string for custom kinds."""

        @trace_function(self.client, name="func", kind="DATA_PROCESSING")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"


class TestTraceFunctionIOCapture:
    """
    trace_function auto-captures the decorated function's inputs
    (positional + keyword args keyed by parameter name) and return
    value onto the span as ``agentic.function.inputs`` /
    ``agentic.function.output``. Both honor the content-capture
    opt-out and support both sync and async functions.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        # Wire a RecordingBuffer so we can introspect what actually
        # got stamped on the span.
        from unittest.mock import MagicMock

        from tests.agentic.instrumentation.conftest import RecordingBuffer

        monkeypatch.setattr("disseqt_agentic_sdk.client.client.HTTPTransport", MagicMock())
        monkeypatch.setattr(
            "disseqt_agentic_sdk.client.client.TraceBuffer",
            lambda **kw: RecordingBuffer(),
        )
        self.client = DisseqtAgenticClient(
            api_key="test_key",
            project_id="test_proj",
            service_name="test_service",
            endpoint="http://localhost/v1/traces",
            application_id="test-app-id",
        )
        yield
        try:
            self.client.shutdown()
        except Exception:
            pass

    def _find_span(self, name: str):
        for span in reversed(self.client.buffer.spans):  # type: ignore[attr-defined]
            if span.name == name:
                return span
        names = [s.name for s in self.client.buffer.spans]  # type: ignore[attr-defined]
        raise AssertionError(f"no span named {name!r}; got {names}")

    def test_captures_positional_and_keyword_args(self):
        import json as _json

        @trace_function(self.client, name="capture_io_kw")
        def echo(user_id, query, limit=10):
            return {"user_id": user_id, "hits": [f"result for {query}"][:limit]}

        result = echo("u_123", query="hello world", limit=5)
        assert result == {"user_id": "u_123", "hits": ["result for hello world"]}

        span = self._find_span("capture_io_kw")
        # Inputs — keyed by parameter name (positional args bound by
        # inspect.signature).
        inputs = _json.loads(_json.loads(span.attributes_json)[AgenticAttributes.FUNCTION_INPUTS])
        assert inputs == {"user_id": "u_123", "query": "hello world", "limit": 5}
        # Output — return value serialized as-is.
        assert _json.loads(
            _json.loads(span.attributes_json)[AgenticAttributes.FUNCTION_OUTPUT]
        ) == {
            "user_id": "u_123",
            "hits": ["result for hello world"],
        }

    def test_capture_io_false_skips_both(self):
        import json as _json

        @trace_function(self.client, name="capture_io_off", capture_io=False)
        def sensitive(secret):
            return "ok"

        sensitive("hunter2")

        span = self._find_span("capture_io_off")
        assert AgenticAttributes.FUNCTION_INPUTS not in _json.loads(span.attributes_json)
        assert AgenticAttributes.FUNCTION_OUTPUT not in _json.loads(span.attributes_json)

    def test_capture_content_off_gates_inputs_and_output(self):
        """
        FUNCTION_INPUTS / FUNCTION_OUTPUT are in _CONTENT_ATTR_KEYS, so
        set_capture_content(False) at deploy time must redact them.
        """
        import json as _json

        from disseqt_agentic_sdk.instrumentation import (
            get_capture_content,
            set_capture_content,
        )

        original = get_capture_content()
        set_capture_content(False)
        try:

            @trace_function(self.client, name="gated_capture")
            def leaky(user_id, api_key):
                return f"charged {user_id} with {api_key}"

            leaky("u_1", api_key="sk_LIVE_secret")
        finally:
            set_capture_content(original)

        span = self._find_span("gated_capture")
        attrs = _json.loads(span.attributes_json)
        payload = _json.dumps(attrs, default=str)
        assert "sk_LIVE_secret" not in payload
        assert "charged" not in payload
        # And the specific keys shouldn't be set at all.
        assert AgenticAttributes.FUNCTION_INPUTS not in attrs
        assert AgenticAttributes.FUNCTION_OUTPUT not in attrs

    def test_async_function_is_traced(self):
        import asyncio
        import json as _json

        @trace_function(self.client, name="async_op")
        async def fetch(url, timeout=5):
            await asyncio.sleep(0)
            return {"url": url, "status": 200}

        result = asyncio.run(fetch("https://example.com", timeout=3))
        assert result == {"url": "https://example.com", "status": 200}

        span = self._find_span("async_op")
        inputs = _json.loads(_json.loads(span.attributes_json)[AgenticAttributes.FUNCTION_INPUTS])
        assert inputs == {"url": "https://example.com", "timeout": 3}
        assert _json.loads(
            _json.loads(span.attributes_json)[AgenticAttributes.FUNCTION_OUTPUT]
        ) == {
            "url": "https://example.com",
            "status": 200,
        }

    def test_exception_still_captures_inputs_and_records_error(self):
        """
        On exception, inputs still land (they were stamped at span
        open) but no output attribute is set. The exception propagates
        and is recorded as span error.
        """
        import json as _json

        @trace_function(self.client, name="raising_fn")
        def blow_up(x):
            raise RuntimeError(f"boom {x}")

        with pytest.raises(RuntimeError, match="boom 42"):
            blow_up(42)

        span = self._find_span("raising_fn")
        inputs = _json.loads(_json.loads(span.attributes_json)[AgenticAttributes.FUNCTION_INPUTS])
        assert inputs == {"x": 42}
        # No output on failure.
        assert AgenticAttributes.FUNCTION_OUTPUT not in _json.loads(span.attributes_json)
        # And the span was marked ERROR.
        assert span.status_code == "ERROR"

    def test_non_serializable_arg_falls_back_gracefully(self):
        """A non-JSON-serializable arg must not crash the decorator."""
        import json as _json

        class Weird:
            def __str__(self):
                return "<Weird instance>"

        @trace_function(self.client, name="weird_arg")
        def take_weird(obj):
            return "ok"

        take_weird(Weird())

        span = self._find_span("weird_arg")
        # Inputs key present; value contains the __str__ fallback.
        inputs_str = _json.loads(span.attributes_json)[AgenticAttributes.FUNCTION_INPUTS]
        assert "Weird" in inputs_str or "<Weird instance>" in inputs_str

    def test_model_exec_kind_stamps_llm_shape_from_str_in_str_out(self):
        """
        When kind=MODEL_EXEC and the function is str-in / str-out, the
        decorator also stamps agentic.input.messages /
        agentic.output.messages / agentic.operation.name so the span
        matches native auto-instrumented provider spans downstream.
        """
        import json as _json

        @trace_function(self.client, kind=SpanKind.MODEL_EXEC, name="my_custom_llm")
        def my_llm(query: str) -> str:
            return f"echo: {query}"

        my_llm("What is the capital of France?")

        span = self._find_span("my_custom_llm")
        attrs = _json.loads(span.attributes_json)

        # LLM-shaped attrs — identical shape to a native OpenAI /
        # Gemini / Anthropic auto-instrumented span.
        assert attrs[AgenticAttributes.INPUT_MESSAGES] == [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        assert attrs[AgenticAttributes.OUTPUT_MESSAGES] == [
            {"role": "assistant", "content": "echo: What is the capital of France?"}
        ]
        assert attrs[AgenticAttributes.OPERATION_NAME] == AgenticOperation.CHAT

        # Generic function.inputs/output still land for the debug view.
        assert AgenticAttributes.FUNCTION_INPUTS in attrs
        assert AgenticAttributes.FUNCTION_OUTPUT in attrs

    def test_non_model_exec_kind_skips_llm_shape(self):
        """kind=INTERNAL keeps only function.inputs/output — no LLM attrs."""
        import json as _json

        @trace_function(self.client, name="plain_step")
        def plain(x: str) -> str:
            return f"plain: {x}"

        plain("hello")

        span = self._find_span("plain_step")
        attrs = _json.loads(span.attributes_json)
        assert AgenticAttributes.INPUT_MESSAGES not in attrs
        assert AgenticAttributes.OUTPUT_MESSAGES not in attrs

    def test_nested_decorated_calls_share_one_trace(self):
        """
        Chaining: when a decorated function calls another decorated
        function, the inner call must nest as a child span under the
        outer trace — NOT open a second top-level trace. Detection
        uses get_current_trace() (thread-local) set by the outer
        start_trace's __enter__.
        """

        @trace_function(self.client, name="inner_step")
        def inner():
            return "inner_result"

        @trace_function(self.client, name="outer_step")
        def outer():
            return inner()

        assert outer() == "inner_result"

        # Both spans should share the same trace_id — proves the inner
        # call opened a child on the outer's trace instead of a new one.
        outer_span = self._find_span("outer_step")
        inner_span = self._find_span("inner_step")
        assert (
            outer_span.trace_id == inner_span.trace_id
        ), f"expected shared trace; outer={outer_span.trace_id} inner={inner_span.trace_id}"

        # And the inner span's parent should be the outer span.
        assert (
            inner_span.parent_span_id == outer_span.span_id
        ), f"inner span parent {inner_span.parent_span_id} != outer {outer_span.span_id}"

    def test_large_payload_is_truncated(self):
        """A pathologically large arg must not be persisted verbatim."""
        import json as _json

        big = "A" * 100_000

        @trace_function(self.client, name="big_arg")
        def take_big(blob):
            return "ok"

        take_big(big)

        span = self._find_span("big_arg")
        inputs_str = _json.loads(span.attributes_json)[AgenticAttributes.FUNCTION_INPUTS]
        # Well under the raw 100_000 chars.
        assert len(inputs_str) < 25_000
        assert "truncated" in inputs_str


class TestClientHelpers:
    """Tests for client helper functions."""

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_get_client_when_initialized(self, mock_trace_buffer, mock_http_transport):
        """Test get_client returns client when initialized."""
        client = DisseqtAgenticClient(
            api_key="test_key",
            project_id="test_proj",
            service_name="test_service",
            application_id="test-app-id",
        )
        set_client(client)

        retrieved_client = get_client()
        assert retrieved_client is not None
        assert retrieved_client.project_id == "test_proj"

        client.shutdown()
        set_client(None)

    def test_get_client_when_not_initialized(self):
        """Test get_client returns None when not initialized."""
        set_client(None)

        client = get_client()
        assert client is None
