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
        with patch("disseqt_agentic_sdk.client.client.HTTPTransport"), patch(
            "disseqt_agentic_sdk.client.client.TraceBuffer"
        ):
            self.client = DisseqtAgenticClient(
                api_key="test_key",
                org_id="test_org",
                project_id="test_proj",
                service_name="test_service",
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


class TestClientHelpers:
    """Tests for client helper functions."""

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_get_client_when_initialized(self, mock_trace_buffer, mock_http_transport):
        """Test get_client returns client when initialized."""
        client = DisseqtAgenticClient(
            api_key="test_key",
            org_id="test_org",
            project_id="test_proj",
            service_name="test_service",
        )
        set_client(client)

        retrieved_client = get_client()
        assert retrieved_client is not None
        assert retrieved_client.org_id == "test_org"
        assert retrieved_client.project_id == "test_proj"

        client.shutdown()
        set_client(None)

    def test_get_client_when_not_initialized(self):
        """Test get_client returns None when not initialized."""
        set_client(None)

        client = get_client()
        assert client is None
