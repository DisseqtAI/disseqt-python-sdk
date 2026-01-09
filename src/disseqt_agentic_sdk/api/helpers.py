"""
Helper functions for easier trace and span creation.

These functions simplify common operations like tracing LLM calls,
agent actions, and tool calls.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

from disseqt_agentic_sdk.client import DisseqtAgenticClient
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.semantics import AgenticAttributes, AgenticOperation


def trace_llm_call(
    trace,
    name: str,
    model_name: str,
    provider: str,
    input_messages: list[dict[str, Any]] | None = None,
    output_messages: list[dict[str, Any]] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
):
    """
    Create an LLM call span with common attributes pre-filled.

    Args:
        trace: DisseqtTrace instance
        name: Span name
        model_name: Model name (e.g., "gpt-4")
        provider: Provider name (e.g., "openai")
        input_messages: Input messages
        output_messages: Output messages
        input_tokens: Input token count
        output_tokens: Output token count
        temperature: Temperature setting
        max_tokens: Max tokens setting
        **kwargs: Additional attributes

    Returns:
        DisseqtSpan: The created span

    Example:
        >>> with start_trace("my_trace") as trace:
        ...     trace_llm_call(
        ...         trace,
        ...         name="chat_completion",
        ...         model_name="gpt-4",
        ...         provider="openai",
        ...         input_tokens=100,
        ...         output_tokens=50
        ...     )
    """
    span = trace.start_span(name, SpanKind.MODEL_EXEC)

    span.set_model_info(model_name, provider)
    span.set_operation(AgenticOperation.CHAT)

    if input_messages:
        span.set_messages(input_messages=input_messages)
    if output_messages:
        span.set_messages(output_messages=output_messages)
    if input_tokens is not None and output_tokens is not None:
        span.set_token_usage(input_tokens, output_tokens)
    if temperature is not None:
        span.set_attribute("agentic.request.temperature", temperature)
    if max_tokens is not None:
        span.set_attribute("agentic.request.max_tokens", max_tokens)

    # Add any additional attributes
    for key, value in kwargs.items():
        span.set_attribute(key, value)

    return span


def trace_agent_action(
    trace,
    name: str,
    agent_name: str,
    agent_id: str | None = None,
    agent_version: str | None = None,
    operation: str | None = None,
    **kwargs,
):
    """
    Create an agent action span with common attributes pre-filled.

    Args:
        trace: DisseqtTrace instance
        name: Span name
        agent_name: Agent name
        agent_id: Optional agent ID
        agent_version: Optional agent version
        operation: Optional operation name
        **kwargs: Additional attributes

    Returns:
        DisseqtSpan: The created span

    Example:
        >>> with start_trace("my_trace") as trace:
        ...     trace_agent_action(
        ...         trace,
        ...         name="planning",
        ...         agent_name="assistant",
        ...         agent_id="agent_001"
        ...     )
    """
    span = trace.start_span(name, SpanKind.AGENT_EXEC)

    span.set_agent_info(agent_name, agent_id, agent_version)
    if operation:
        span.set_operation(operation)

    # Add any additional attributes
    for key, value in kwargs.items():
        span.set_attribute(key, value)

    return span


def trace_tool_call(
    trace,
    name: str,
    tool_name: str,
    call_id: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
    **kwargs,
):
    """
    Create a tool call span with common attributes pre-filled.

    Args:
        trace: DisseqtTrace instance
        name: Span name
        tool_name: Tool name
        call_id: Optional call ID
        tool_definitions: Optional tool definitions
        **kwargs: Additional attributes

    Returns:
        DisseqtSpan: The created span

    Example:
        >>> with start_trace("my_trace") as trace:
        ...     trace_tool_call(
        ...         trace,
        ...         name="weather_api",
        ...         tool_name="get_weather",
        ...         call_id="call_001"
        ...     )
    """
    span = trace.start_span(name, SpanKind.TOOL_EXEC)

    span.set_tool_info(tool_name, call_id)
    if tool_definitions:
        span.set_attribute(AgenticAttributes.TOOL_DEFINITIONS, tool_definitions)
    span.set_operation(AgenticOperation.EXECUTE_TOOL)

    # Add any additional attributes
    for key, value in kwargs.items():
        span.set_attribute(key, value)

    return span


def trace_function(
    client: DisseqtAgenticClient,
    name: str | None = None,
    kind: SpanKind | str = SpanKind.INTERNAL,
    **span_attrs,
):
    """
    Decorator to automatically trace a function.

    Args:
        client: DisseqtAgenticClient instance (required)
        name: Optional span name (defaults to function name)
        kind: Span kind (default: INTERNAL)
        **span_attrs: Additional span attributes

    Example:
        >>> client = DisseqtAgenticClient(...)
        >>> @trace_function(client, name="my_function")
        ... def my_function():
        ...     return "result"

        >>> @trace_function(client, kind=SpanKind.AGENT_EXEC, agent_name="assistant")
        ... def agent_function():
        ...     return "result"
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from disseqt_agentic_sdk.api.trace import start_trace

            span_name = name or func.__name__

            # Convert string to SpanKind if needed
            span_kind = SpanKind(kind) if isinstance(kind, str) else kind

            with start_trace(client, f"{span_name}_trace") as trace:
                with trace.start_span(span_name, span_kind) as span:
                    # Set any provided attributes
                    for key, value in span_attrs.items():
                        span.set_attribute(key, value)

                    # Execute function
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        span.set_error(str(e), error_type=type(e).__name__)
                        raise

        return wrapper

    return decorator
