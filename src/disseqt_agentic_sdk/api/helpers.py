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
from disseqt_agentic_sdk.instrumentation._utils import (
    safe_set,
    set_messages_if_capturing,
)
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
    realtime_policy_id: str | None = None,
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
        realtime_policy_id: Optional per-span policy override. When set,
            this span is validated against this policy instead of the
            trace-level or client-level default.
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
    span = trace.start_span(name, SpanKind.MODEL_EXEC, realtime_policy_id=realtime_policy_id)

    span.set_model_info(model_name, provider)
    span.set_operation(AgenticOperation.CHAT)

    # Route message-body writes through the content-capture gate so
    # set_capture_content(False) actually redacts on the manual API
    # too — the auto-instrumentation path always did, this public
    # helper used to bypass it (TP-2128 round-2 P0 #0.2).
    if input_messages:
        set_messages_if_capturing(span, input_messages=input_messages)
    if output_messages:
        set_messages_if_capturing(span, output_messages=output_messages)
    if input_tokens is not None and output_tokens is not None:
        span.set_token_usage(input_tokens, output_tokens)
    if temperature is not None:
        span.set_attribute("agentic.request.temperature", temperature)
    if max_tokens is not None:
        span.set_attribute("agentic.request.max_tokens", max_tokens)

    # Route caller-supplied kwargs through safe_set so any content-
    # attribute keys the caller passes (e.g. AgenticAttributes.PROMPT
    # via `**{AgenticAttributes.PROMPT: ...}`) also honor the gate.
    for key, value in kwargs.items():
        safe_set(span, key, value)

    return span


def trace_agent_action(
    trace,
    name: str,
    agent_name: str,
    agent_id: str | None = None,
    agent_version: str | None = None,
    operation: str | None = None,
    realtime_policy_id: str | None = None,
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
        realtime_policy_id: Optional per-span policy override. When set,
            this span is validated against this policy instead of the
            trace-level or client-level default.
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
    span = trace.start_span(name, SpanKind.AGENT_EXEC, realtime_policy_id=realtime_policy_id)

    span.set_agent_info(agent_name, agent_id, agent_version)
    if operation:
        span.set_operation(operation)

    # safe_set honors the content-capture gate for content-shaped keys
    # and skips None / empty values (TP-2128 round-2 P0 #0.2).
    for key, value in kwargs.items():
        safe_set(span, key, value)

    return span


def trace_tool_call(
    trace,
    name: str,
    tool_name: str,
    call_id: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
    realtime_policy_id: str | None = None,
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
        realtime_policy_id: Optional per-span policy override. When set,
            this span is validated against this policy instead of the
            trace-level or client-level default.
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
    span = trace.start_span(name, SpanKind.TOOL_EXEC, realtime_policy_id=realtime_policy_id)

    span.set_tool_info(tool_name, call_id)
    if tool_definitions:
        # Gate tool_definitions specifically — schemas can carry
        # credential-shaped defaults (e.g. send_email tool with a
        # smtp_password parameter). TOOL_DEFINITIONS is in
        # _CONTENT_ATTR_KEYS (round-2 P1 #1.1) so safe_set honors
        # the capture toggle.
        safe_set(span, AgenticAttributes.TOOL_DEFINITIONS, tool_definitions)
    span.set_operation(AgenticOperation.EXECUTE_TOOL)

    # safe_set honors the content-capture gate + None/empty skip
    # (TP-2128 round-2 P0 #0.2).
    for key, value in kwargs.items():
        safe_set(span, key, value)

    return span


def trace_function(
    client: DisseqtAgenticClient,
    name: str | None = None,
    kind: SpanKind | str = SpanKind.INTERNAL,
    realtime_policy_id: str | None = None,
    trace_realtime_policy_id: str | None = None,
    **span_attrs,
):
    """
    Decorator to automatically trace a function.

    Args:
        client: DisseqtAgenticClient instance (required)
        name: Optional span name (defaults to function name)
        kind: Span kind (default: INTERNAL). Can be a SpanKind enum value or custom string.
        realtime_policy_id: Optional per-span policy override applied to the
            wrapped function's span. When set, this span is validated against
            this policy instead of the trace-level or client-level default.
        trace_realtime_policy_id: Optional per-trace policy override applied to
            the auto-created trace. Use this when you want the whole trace
            (including any child spans opened inside the wrapped function) to
            run under a specific policy.
        **span_attrs: Additional span attributes

    Example:
        >>> client = DisseqtAgenticClient(..., application_id="...")
        >>> @trace_function(client, name="my_function")
        ... def my_function():
        ...     return "result"

        >>> @trace_function(client, kind=SpanKind.AGENT_EXEC, agent_name="assistant")
        ... def agent_function():
        ...     return "result"

        >>> @trace_function(client, kind="CUSTOM_OPERATION")
        ... def custom_function():
        ...     return "result"

        >>> @trace_function(client, realtime_policy_id="policy-uuid-for-this-span")
        ... def guarded_function():
        ...     return "result"
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from disseqt_agentic_sdk.api.trace import start_trace

            span_name = name or func.__name__

            # Convert string to SpanKind if it's a valid enum value, otherwise keep as string for custom kinds
            if isinstance(kind, str):
                try:
                    span_kind = SpanKind(kind)
                except ValueError:
                    # Custom span kind - keep as string
                    span_kind = kind
            else:
                span_kind = kind

            with start_trace(
                client,
                f"{span_name}_trace",
                realtime_policy_id=trace_realtime_policy_id,
            ) as trace:
                with trace.start_span(
                    span_name, span_kind, realtime_policy_id=realtime_policy_id
                ) as span:
                    # safe_set honors the content-capture gate for
                    # any content-shaped keys passed via **span_attrs
                    # (TP-2128 round-2 P0 #0.2).
                    for key, value in span_attrs.items():
                        safe_set(span, key, value)

                    # Execute function
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        span.set_error(str(e), error_type=type(e).__name__)
                        raise

        return wrapper

    return decorator
