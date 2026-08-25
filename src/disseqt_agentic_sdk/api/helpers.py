"""
Helper functions for easier trace and span creation.

These functions simplify common operations like tracing LLM calls,
agent actions, and tool calls.
"""

import asyncio
import contextlib
import inspect
import json
from collections.abc import Callable, Iterator
from functools import wraps
from typing import Any

from disseqt_agentic_sdk.client import DisseqtAgenticClient
from disseqt_agentic_sdk.context import get_current_trace
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._utils import (
    safe_set,
    set_messages_if_capturing,
)
from disseqt_agentic_sdk.semantics import AgenticAttributes, AgenticOperation


def _extract_llm_input_messages(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> list[dict[str, str]] | None:
    """
    Best-effort turn a ``kind=MODEL_EXEC`` function's args into an
    OpenAI-shaped ``input_messages`` list so the span matches native
    auto-instrumented provider spans (dashboards + validators keying
    on ``agentic.input.messages`` work unchanged).

    Priority: ``messages`` (already list) → ``prompt`` / ``query`` /
    ``question`` / ``input`` (str) → first string param. Returns None
    when nothing matches — caller falls back to generic function.inputs
    only (no LLM-shape attrs stamped).
    """
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        params = bound.arguments
    except (TypeError, ValueError):
        return None

    val = params.get("messages")
    if isinstance(val, list):
        return val  # already OpenAI chat shape — pass through unchanged

    for name in ("prompt", "query", "question", "input"):
        val = params.get(name)
        if isinstance(val, str) and val:
            return [{"role": "user", "content": val}]

    # Fallback: first string-valued arg (respects call order via
    # inspect.signature.bind_partial keeping insertion order).
    for val in params.values():
        if isinstance(val, str) and val:
            return [{"role": "user", "content": val}]
    return None


def _extract_tool_args(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Bind ``args``/``kwargs`` back to the function signature so tool
    invocations record a dict of ``{param_name: value}`` on
    ``agentic.tool.args`` — matching the shape auto-instrumented
    provider tool_call spans produce.
    """
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return None
    if not bound.arguments:
        return None
    return dict(bound.arguments)


def _json_safe(value: Any) -> Any:
    """
    Coerce arbitrary tool return values into something the span's
    ``attributes_json`` serializer can handle. Pass primitives / lists
    / dicts through; fall back to ``str()`` for objects that don't
    round-trip through ``json.dumps`` (pydantic models, ORM rows,
    custom classes). Never raises — worst case yields ``repr()``.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        try:
            return str(value)
        except Exception:
            return repr(value)


def _extract_llm_output_messages(
    result: Any,
) -> list[dict[str, str]] | None:
    """
    Best-effort turn a ``kind=MODEL_EXEC`` function's return value
    into an OpenAI-shaped ``output_messages`` list.

    Recognizes plain str, OpenAI ChatCompletion (``choices[0].message.
    content``), Anthropic Message (``content[0].text``), and Gemini
    GenerateContentResponse (``candidates[0].content.parts[0].text``)
    — the four shapes native auto-instrumented providers already
    produce. Returns None on anything else — caller falls back to
    generic function.output only.
    """
    if isinstance(result, str) and result:
        return [{"role": "assistant", "content": result}]

    # Duck-type into common provider response shapes. Wrap in try to
    # never crash observability on an unexpected wire shape.
    try:
        # OpenAI-shape (dict or pydantic-with-attrs)
        choices = _read_attr_or_key(result, "choices")
        if choices:
            first = choices[0]
            message = _read_attr_or_key(first, "message")
            content = _read_attr_or_key(message, "content")
            if isinstance(content, str) and content:
                return [{"role": "assistant", "content": content}]

        # Anthropic-shape
        content_blocks = _read_attr_or_key(result, "content")
        if isinstance(content_blocks, list) and content_blocks:
            first_block = content_blocks[0]
            text = _read_attr_or_key(first_block, "text")
            if isinstance(text, str) and text:
                return [{"role": "assistant", "content": text}]

        # Gemini-shape
        candidates = _read_attr_or_key(result, "candidates")
        if candidates:
            first_cand = candidates[0]
            cand_content = _read_attr_or_key(first_cand, "content")
            parts = _read_attr_or_key(cand_content, "parts")
            if parts:
                text = _read_attr_or_key(parts[0], "text")
                if isinstance(text, str) and text:
                    return [{"role": "assistant", "content": text}]
    except (AttributeError, TypeError, IndexError, KeyError):
        pass
    return None


def _read_attr_or_key(obj: Any, name: str) -> Any:
    """Tolerant attr-or-key read used by the LLM-shape duck-typers."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


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


def disseqt_trace(
    client: DisseqtAgenticClient | Callable[..., Any] | None = None,
    name: str | None = None,
    kind: SpanKind | str = SpanKind.INTERNAL,
    realtime_policy_id: str | None = None,
    trace_realtime_policy_id: str | None = None,
    capture_io: bool = True,
    **span_attrs,
):
    """
    Decorator to automatically trace a function.

    When ``kind=SpanKind.MODEL_EXEC``, auto-captures the function's
    LLM-shaped inputs and output onto the span as
    ``agentic.input.messages`` / ``agentic.output.messages`` /
    ``agentic.operation.name`` — the same attribute keys native
    auto-instrumented providers produce, so validators / dashboards
    treat a decorated custom LLM identically to a real provider call.

    When ``kind=SpanKind.TOOL_EXEC``, stamps tool-shape attrs so the
    span is indistinguishable from a tool_call span the auto-
    instrumentors already produce: ``agentic.tool.name`` (the function
    name or the ``name=`` decorator arg — this one drives the
    ``is_tool_call`` server-side flag, since it's computed as
    ``tool_name IS NOT NULL``), ``agentic.tool.args`` (the bound
    kwargs), ``agentic.tool.result`` (the return value), and
    ``agentic.operation.name = "execute_tool"``. Non-JSON-serializable
    return values are coerced to ``str()`` so a weird tool response
    doesn't fail the span send.

    Other span kinds get just the span itself (no I/O attributes).
    ``agentic.input.messages``, ``agentic.output.messages``,
    ``agentic.tool.args`` and ``agentic.tool.result`` all honor the
    content-capture opt-out (``set_capture_content(False)`` /
    ``DISSEQT_SDK_CAPTURE_CONTENT=0``); pass ``capture_io=False`` to
    skip content stamping entirely (tool.name and operation.name stay
    since they're not content).

    Auto-chains into a parent-child span tree: when a decorated
    function calls another decorated function, the inner call opens a
    child span on the outer trace instead of creating its own
    top-level trace. Detection is via ``get_current_trace()`` (thread-
    local) — a nested call sees the outer trace and nests under it;
    a top-level call sees nothing and bootstraps its own trace.

    Client resolution:
      * If ``client=`` is passed explicitly → used directly.
      * Otherwise resolved at call-time from ``get_client()`` (the
        process-default client set by the most recent
        ``DisseqtAgenticClient(...)`` construction — the constructor
        auto-registers itself). Raises ``RuntimeError`` if no client
        has been constructed yet.

    Usage forms:
      * ``@disseqt_trace`` — bare, no parens. Uses defaults + default
        client. Cheapest ergonomic form.
      * ``@disseqt_trace(kind=SpanKind.MODEL_EXEC, name="my_llm")`` —
        parametrized, still uses default client.
      * ``@disseqt_trace(client=my_client, ...)`` — explicit client
        (only needed when running multiple clients in one process).

    Supports both sync and async functions — the wrapper is chosen
    automatically based on the decorated callable.

    Args:
        client: Optional DisseqtAgenticClient. When None, resolved via
            ``get_client()`` at call time (the constructor auto-
            registers itself as the default). Explicit for multi-client
            deployments.
        name: Optional span name (defaults to function name)
        kind: Span kind (default: INTERNAL). Can be a SpanKind enum value or custom string.
        realtime_policy_id: Optional per-span policy override applied to the
            wrapped function's span. When set, this span is validated against
            this policy instead of the trace-level or client-level default.
        trace_realtime_policy_id: Optional per-trace policy override applied to
            the auto-created trace. Use this when you want the whole trace
            (including any child spans opened inside the wrapped function) to
            run under a specific policy.
        capture_io: When True (default), stamps the LLM-shaped
            attributes on a MODEL_EXEC span or the tool-shape
            ``agentic.tool.args`` / ``agentic.tool.result`` on a
            TOOL_EXEC span. Set False to skip content stamping on a
            span whose args or return value aren't safe/useful to
            record. TOOL_EXEC's ``tool.name`` and
            ``operation.name=execute_tool`` are always stamped since
            they aren't content and the server derives
            ``is_tool_call`` from ``tool.name``.
        **span_attrs: Additional span attributes stamped once at span open.

    Example:
        >>> client = DisseqtAgenticClient(..., application_id="...")

        >>> # Simplest — bare decorator, uses defaults + default client.
        >>> @disseqt_trace
        ... def my_step(x): return x + 1

        >>> # LLM-shaped span, no client passed — resolves via get_client().
        >>> @disseqt_trace(kind=SpanKind.MODEL_EXEC, name="my_llm")
        ... def my_llm(query: str) -> str: return call_my_model(query)

        >>> # Explicit client override — multi-client deployments.
        >>> @disseqt_trace(client=other_client, kind=SpanKind.AGENT_EXEC)
        ... def agent_step(): ...
    """
    # Bare-decorator detection: ``@disseqt_trace`` (no parens) makes
    # Python call ``disseqt_trace(the_decorated_function)`` — so the
    # ``client`` positional slot actually holds a plain callable. Wrap
    # + apply immediately with ``client=None`` (resolved at call time
    # via get_client()). This matches the ergonomic bare-decorator
    # form of comparable tracing SDKs.
    if callable(client) and not isinstance(client, DisseqtAgenticClient):
        func_being_decorated = client
        resolved_client: DisseqtAgenticClient | None = None
    else:
        func_being_decorated = None
        resolved_client = client

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
        # When kind=MODEL_EXEC we stamp the LLM-shaped attrs
        # (agentic.input.messages / agentic.output.messages /
        # agentic.operation.name) so a decorated custom LLM call is
        # indistinguishable from a native auto-instrumented provider
        # call for every downstream validator / dashboard. Other span
        # kinds get just the span itself — no I/O attribute pollution.
        is_llm_kind = span_kind == SpanKind.MODEL_EXEC
        # kind=TOOL_EXEC stamps tool-shape attrs so the span looks
        # identical to a tool_call span produced by native auto-
        # instrumentation. In particular ``agentic.tool.name`` is what
        # the backend enricher extracts into the ``tool_name`` column;
        # the ``is_tool_call`` flag is computed as
        # ``tool_name IS NOT NULL``, so without this the flag stays
        # false even on a TOOL_EXEC span.
        is_tool_kind = span_kind == SpanKind.TOOL_EXEC

        def _prep_span(span: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            """Stamp static + I/O attributes at span open."""
            # Static attrs — honor content-capture gate for content-shaped keys
            # (TP-2128 round-2 P0 #0.2).
            for key, value in span_attrs.items():
                safe_set(span, key, value)
            if capture_io and is_llm_kind:
                safe_set(span, AgenticAttributes.OPERATION_NAME, AgenticOperation.CHAT)
                input_messages = _extract_llm_input_messages(func, args, kwargs)
                if input_messages is not None:
                    set_messages_if_capturing(span, input_messages=input_messages)
            elif is_tool_kind:
                # tool.name is unconditional — it drives ``is_tool_call``
                # server-side. Only tool.args goes through the content
                # gate (it can carry PII / credentials).
                safe_set(span, AgenticAttributes.TOOL_NAME, span_name_default)
                safe_set(span, AgenticAttributes.OPERATION_NAME, AgenticOperation.EXECUTE_TOOL)
                if capture_io:
                    tool_args = _extract_tool_args(func, args, kwargs)
                    if tool_args is not None:
                        safe_set(
                            span,
                            AgenticAttributes.TOOL_ARGS,
                            {k: _json_safe(v) for k, v in tool_args.items()},
                        )

        def _record_output(span: Any, result: Any) -> None:
            if capture_io and is_llm_kind:
                output_messages = _extract_llm_output_messages(result)
                if output_messages is not None:
                    set_messages_if_capturing(span, output_messages=output_messages)
            elif capture_io and is_tool_kind and result is not None:
                # safe_set gates TOOL_RESULT via _CONTENT_ATTR_KEYS, so
                # set_capture_content(False) redacts tool returns just
                # like it redacts message content. _json_safe coerces
                # non-serializable return values (pydantic models, ORM
                # rows) into a string so a weird tool return doesn't
                # crash the whole span send at json.dumps time.
                safe_set(span, AgenticAttributes.TOOL_RESULT, _json_safe(result))

        @contextlib.contextmanager
        def _open_span() -> Iterator[Any]:
            """
            Yield the span this call should populate — nested under an
            active trace if one exists, otherwise wrapped in a freshly
            bootstrapped trace.

            Chaining: when a decorated function calls another decorated
            function, ``get_current_trace()`` returns the outer trace
            (set by the outer ``start_trace``'s ``__enter__``), so the
            inner call opens a *child span* on it instead of opening a
            second top-level trace. Match the parent-child structure
            other tracing SDKs produce via ContextVar-based run
            detection — see the ``@disseqt_trace`` docstring for a
            waterfall example.
            """
            from disseqt_agentic_sdk.api.client import get_client
            from disseqt_agentic_sdk.api.trace import start_trace

            active = get_current_trace()
            if active is not None:
                # Nested call — inherit the outer trace; client not
                # needed here (the outer trace already knows its client).
                with active.start_span(
                    span_name_default,
                    span_kind,
                    realtime_policy_id=realtime_policy_id,
                ) as span:
                    yield span
                return

            # Top-level call — resolve the client to use. Prefer the
            # explicit one passed at decoration time; fall back to the
            # process-default set by ``DisseqtAgenticClient.__init__``.
            # Resolved at CALL time (not decoration time) so decorators
            # applied at import — before the client is constructed —
            # still work.
            call_time_client = resolved_client or get_client()
            if call_time_client is None:
                raise RuntimeError(
                    "@disseqt_trace has no client to open a trace with. "
                    "Either construct a DisseqtAgenticClient(...) before "
                    "calling the decorated function (the constructor auto-"
                    "registers as the process default), or pass "
                    "client=... to the decorator explicitly."
                )
            with start_trace(
                call_time_client,
                f"{span_name_default}_trace",
                realtime_policy_id=trace_realtime_policy_id,
            ) as trace:
                with trace.start_span(
                    span_name_default,
                    span_kind,
                    realtime_policy_id=realtime_policy_id,
                ) as span:
                    yield span

        if is_coro:

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _open_span() as span:
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
            with _open_span() as span:
                _prep_span(span, args, kwargs)
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    span.set_error(str(e), error_type=type(e).__name__)
                    raise
                _record_output(span, result)
                return result

        return sync_wrapper

    # Bare-decorator (@disseqt_trace without parens): apply immediately.
    if func_being_decorated is not None:
        return decorator(func_being_decorated)
    return decorator
