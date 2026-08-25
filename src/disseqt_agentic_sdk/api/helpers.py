"""
Helper functions for easier trace and span creation.

These functions simplify common operations like tracing LLM calls,
agent actions, and tool calls.
"""

import asyncio
import inspect
import json
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

# Maximum characters per auto-captured I/O value. Truncates the JSON
# string on the span if the caller's args or return value are absurdly
# large (a 50MB payload, a numpy array's full repr, etc.). Chosen large
# enough to not lose typical chat / RAG payloads (usually 5–20 KB),
# small enough to keep a single span from bloating the wire request.
_TRACE_FUNCTION_MAX_IO_CHARS = 20_000


def _serialize_for_span(value: Any) -> str:
    """
    Best-effort JSON-serialize a value for storage on a span attribute.

    * ``json.dumps(..., default=str)`` covers pydantic models
      (via ``model_dump`` if the model overrides ``__str__``), datetime,
      Decimal, UUID, and everything with a sane ``__str__``.
    * Anything json still refuses (open sockets, thread handles, C
      extension objects) falls back to ``repr(value)``.
    * Result is truncated to ``_TRACE_FUNCTION_MAX_IO_CHARS`` chars so a
      pathologically large arg can't blow up a span.
    """
    try:
        s = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(value)
    if len(s) > _TRACE_FUNCTION_MAX_IO_CHARS:
        s = s[:_TRACE_FUNCTION_MAX_IO_CHARS] + "…[truncated]"
    return s


def _build_inputs_payload(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str:
    """
    Turn a function's positional + keyword args into a stable JSON dict
    keyed by the parameter name. Falls back to ``args[i]`` labels for
    positional args that inspect can't bind (e.g. ``*args`` overflow).

    Named-keys shape means dashboards can index into individual params
    (``inputs.user_id``, ``inputs.query``, ...) without re-parsing.
    """
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        # Don't `apply_defaults()` — capturing the caller's actual
        # inputs (not "what defaults would have filled in") matches
        # LangSmith and keeps the span faithful to what the caller
        # really passed.
        payload = dict(bound.arguments)
    except (TypeError, ValueError):
        # inspect.signature can raise on some C-extension callables;
        # fall back to the raw shape rather than crashing observability.
        payload = {
            **{f"args[{i}]": v for i, v in enumerate(args)},
            **kwargs,
        }
    return _serialize_for_span(payload)


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
    capture_io: bool = True,
    **span_attrs,
):
    """
    Decorator to automatically trace a function.

    Auto-captures the function's inputs (positional + keyword args, keyed
    by parameter name) and its return value onto the span as
    ``agentic.function.inputs`` and ``agentic.function.output``. Both
    values honor the content-capture opt-out (``set_capture_content(
    False)`` / ``DISSEQT_SDK_CAPTURE_CONTENT=0``) since function args
    can carry credentials or PII. Pass ``capture_io=False`` to skip.

    Supports both sync and async functions — the wrapper is chosen
    automatically based on the decorated callable.

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
        capture_io: When True (default) auto-captures function args + return
            value onto ``agentic.function.inputs`` / ``agentic.function.output``.
            Set False for functions whose args are large binary blobs, non-
            serializable handles, or otherwise not useful to record.
        **span_attrs: Additional span attributes stamped once at span open.

    Example:
        >>> client = DisseqtAgenticClient(..., application_id="...")
        >>> @trace_function(client, name="my_function")
        ... def my_function(user_id: str, query: str) -> str:
        ...     return "result"
        >>> my_function("u_123", query="hello")
        # Span carries:
        #   agentic.function.inputs = '{"user_id": "u_123", "query": "hello"}'
        #   agentic.function.output = '"result"'

        >>> @trace_function(client, kind=SpanKind.MODEL_EXEC)
        ... async def call_custom_llm(prompt: str) -> str:
        ...     return await my_http_llm(prompt)

        >>> @trace_function(client, kind=SpanKind.AGENT_EXEC, agent_name="assistant")
        ... def agent_function(): ...

        >>> @trace_function(client, capture_io=False)  # skip I/O capture
        ... def bulk_ingest(large_df): ...
    """

    def decorator(func: Callable) -> Callable:
        # Convert string to SpanKind if it's a valid enum value,
        # otherwise keep as string for custom kinds. Done once at
        # decoration time (not on every call) since kind is static.
        # SpanKind is a str Enum so ``isinstance(kind, str)`` is always
        # True at runtime — the else branch is defensive for any future
        # kind type that isn't a str subclass. mypy flags it as
        # unreachable given today's SpanKind definition; ignore rather
        # than delete so a signature widening doesn't lose the guard.
        if isinstance(kind, str):
            try:
                span_kind: SpanKind | str = SpanKind(kind)
            except ValueError:
                span_kind = kind
        else:
            span_kind = kind  # type: ignore[unreachable]

        span_name_default = name or func.__name__
        is_coro = asyncio.iscoroutinefunction(func)

        def _prep_span(span: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            """Stamp static + I/O attributes at span open."""
            # Static attrs — honor content-capture gate for content-shaped keys
            # (TP-2128 round-2 P0 #0.2).
            for key, value in span_attrs.items():
                safe_set(span, key, value)
            if capture_io:
                safe_set(
                    span,
                    AgenticAttributes.FUNCTION_INPUTS,
                    _build_inputs_payload(func, args, kwargs),
                )

        def _record_output(span: Any, result: Any) -> None:
            if capture_io:
                safe_set(
                    span,
                    AgenticAttributes.FUNCTION_OUTPUT,
                    _serialize_for_span(result),
                )

        if is_coro:

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                from disseqt_agentic_sdk.api.trace import start_trace

                with start_trace(
                    client,
                    f"{span_name_default}_trace",
                    realtime_policy_id=trace_realtime_policy_id,
                ) as trace:
                    with trace.start_span(
                        span_name_default,
                        span_kind,
                        realtime_policy_id=realtime_policy_id,
                    ) as span:
                        _prep_span(span, args, kwargs)
                        try:
                            result = await func(*args, **kwargs)
                        except Exception as e:
                            span.set_error(str(e), error_type=type(e).__name__)
                            raise
                        _record_output(span, result)
                        return result

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            from disseqt_agentic_sdk.api.trace import start_trace

            with start_trace(
                client,
                f"{span_name_default}_trace",
                realtime_policy_id=trace_realtime_policy_id,
            ) as trace:
                with trace.start_span(
                    span_name_default,
                    span_kind,
                    realtime_policy_id=realtime_policy_id,
                ) as span:
                    _prep_span(span, args, kwargs)
                    try:
                        result = func(*args, **kwargs)
                    except Exception as e:
                        span.set_error(str(e), error_type=type(e).__name__)
                        raise
                    _record_output(span, result)
                    return result

        return sync_wrapper

    return decorator
