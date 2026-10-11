"""
Google ADK patch functions.

Each wrapper is deadlock-safe by construction:

  * Async-generator wrappers (Runner.run_async, BaseAgent.run_async,
    BaseLlm.generate_content_async) call the wrapped function
    synchronously (async-gen functions return an async_generator object
    without executing the body), then hand the generator to
    ``AsyncStreamWrapper`` which handles per-chunk callbacks, cancellation,
    exhaustion, and forwards ``aclose()`` — all guarded by the shared
    ``_closed`` idempotency flag. No await is held across a lock.
  * Async coroutine wrappers (BaseTool.run_async) close the scope in a
    ``try/except BaseException`` so cancellation and ``GeneratorExit``
    still finalize the span; the coroutine is awaited exactly once.
  * All request/response extraction goes through ``safe_call`` / ``read``
    which never propagate exceptions into the caller.
  * No ``asyncio.run`` / ``loop.run_until_complete`` — those would deadlock
    when the wrapper runs inside an already-running event loop (a known
    ADK footgun, see google/adk-python#755).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._stream import AsyncStreamWrapper
from disseqt_agentic_sdk.instrumentation._tool_result import (
    _current_agg,
    _notify_planned_tool_calls,
    _ToolCallAggregator,
)
from disseqt_agentic_sdk.instrumentation._utils import (
    open_llm_span,
    read,
    safe_call,
    safe_set,
    set_first_tool_call_attrs,
    set_messages_if_capturing,
)
from disseqt_agentic_sdk.semantics import (
    AgenticAttributes,
    AgenticOperation,
    AgenticProvider,
    GenAIAttributes,
    GenAIOperation,
    GenAISystem,
)

if TYPE_CHECKING:
    from disseqt_agentic_sdk.instrumentation._utils import _SpanScope
    from disseqt_agentic_sdk.instrumentation.adk.instrumentor import AdkInstrumentor
    from disseqt_agentic_sdk.span import DisseqtSpan


PROVIDER = AgenticProvider.GOOGLE


# ---------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------
def _detect_provider(instance: Any, llm_request: Any) -> tuple[str, str]:
    """Return (agentic_provider, gen_ai_system) for an ADK BaseLlm call.

    ADK can route LLM calls through google-genai (native Gemini) or
    LiteLLM, which itself fronts OpenAI / Anthropic / Cohere / Mistral /
    etc. Previously we hardcoded ("google", "gemini") which skewed
    pricing + model lookups whenever the model was actually OpenAI
    or Claude. Prefer the model string's LiteLLM-style provider prefix,
    fall back to the class name of the ``BaseLlm`` subclass (``Gemini``,
    ``LiteLlm``, ``AnthropicLlm``, …), fall back to Gemini for backward
    compat.
    """
    model = read(llm_request, "model") or ""
    if isinstance(model, str) and "/" in model:
        prefix = model.split("/", 1)[0].lower()
        if prefix in {"openai", "azure", "openrouter"}:
            return (AgenticProvider.OPENAI, GenAISystem.OPENAI)
        if prefix in {"anthropic", "claude"}:
            return (AgenticProvider.ANTHROPIC, GenAISystem.ANTHROPIC)
        if prefix in {"gemini", "google", "vertex_ai", "vertex"}:
            return (AgenticProvider.GOOGLE, GenAISystem.GEMINI)
        # Any other LiteLLM-known prefix (bedrock, cohere, mistral, …):
        # tag it as LiteLLM so backend pricing doesn't apply Gemini
        # rates, and the response.model carries the underlying backend.
        return (AgenticProvider.LITELLM, GenAISystem.LITELLM)
    cls_name = type(instance).__name__.lower() if instance is not None else ""
    if "litellm" in cls_name:
        return (AgenticProvider.LITELLM, GenAISystem.LITELLM)
    if "anthropic" in cls_name or "claude" in cls_name:
        return (AgenticProvider.ANTHROPIC, GenAISystem.ANTHROPIC)
    if "openai" in cls_name or "gpt" in cls_name:
        return (AgenticProvider.OPENAI, GenAISystem.OPENAI)
    # Default to Google/Gemini — the common path through ADK — so a
    # genuine Gemini call doesn't silently lose its provider tag when
    # the model string is a bare short name like "gemini-2.0-flash".
    return (AgenticProvider.GOOGLE, GenAISystem.GEMINI)


# ---------------------------------------------------------------------
# Content normalization
# ---------------------------------------------------------------------
def _normalize_contents(contents: Any) -> list[dict[str, Any]]:
    """
    Coerce an ADK ``LlmRequest.contents`` value into
    ``[{role, content}, ...]`` dicts for span capture.

    ADK routes ``google.genai.types.Content`` objects (``.parts`` +
    ``.role``) into the model layer; a request may also carry bare
    strings or bare ``Part`` objects during construction. Mirrors the
    Gemini instrumentor's normalizer so a call that goes through ADK
    and one that goes straight to google-genai produce identical
    input-message shapes on the span.
    """
    if contents is None:
        return []
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    if isinstance(contents, list):
        return [_normalize_content_entry(e) for e in contents]
    return [_normalize_content_entry(contents)]


def _normalize_content_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        return {"role": "user", "content": entry}
    parts = read(entry, "parts")
    if parts is not None:
        text_pieces: list[str] = []
        for p in parts:
            text = read(p, "text")
            if isinstance(text, str) and text:
                text_pieces.append(text)
        return {"role": read(entry, "role") or "user", "content": "".join(text_pieces)}
    text = read(entry, "text")
    if isinstance(text, str) and text:
        return {"role": "user", "content": text}
    return {"role": "user", "content": str(entry)}


def _extract_content_text(content: Any) -> str:
    """Pull concatenated text from an ``LlmResponse.content`` (a Content object)."""
    if content is None:
        return ""
    parts = read(content, "parts") or []
    pieces: list[str] = []
    for p in parts:
        text = read(p, "text")
        if isinstance(text, str) and text:
            pieces.append(text)
    return "".join(pieces)


def _extract_tool_calls(content: Any, response_id: str | None) -> list[dict[str, Any]]:
    """
    Pull function_call blocks out of an ``LlmResponse.content`` into the
    canonical ``{id, name, arguments}`` shape.

    We deliberately don't route this through ``_tool_calls.from_gemini``
    because ADK's content object is single (not a list of candidates),
    and reusing the Gemini adapter would require synthesizing a
    candidates wrapper. Inlining keeps the two paths independent —
    ADK's LlmResponse shape can drift without breaking Gemini.
    """
    if content is None:
        return []
    parts = read(content, "parts") or []
    calls: list[dict[str, Any]] = []
    for idx, part in enumerate(parts):
        fc = read(part, "function_call")
        if fc is None:
            continue
        name = read(fc, "name")
        if not isinstance(name, str) or not name:
            continue
        call_id = read(fc, "id")
        if not call_id:
            # Synthesize a stable-per-response id so aggregator merging works.
            call_id = f"{response_id or 'adk'}-{idx}"
        args = read(fc, "args")
        if args is None:
            args_str = "{}"
        else:
            try:
                args_str = json.dumps(args, default=str)
            except (TypeError, ValueError):
                args_str = str(args)
        calls.append({"id": str(call_id), "name": name, "arguments": args_str})
    return calls


# ---------------------------------------------------------------------
# LLM request/response attribute writers
# ---------------------------------------------------------------------
def _set_llm_request_attrs(
    span: DisseqtSpan, llm_request: Any, stream: bool, instance: Any = None
) -> None:
    model = read(llm_request, "model") or ""
    agentic_provider, gen_ai_system = _detect_provider(instance, llm_request)
    span.set_model_info(model, agentic_provider)
    span.set_operation(AgenticOperation.GENERATE_CONTENT)
    # Provider derived from the model string (LiteLLM-style prefix) or
    # the BaseLlm subclass. A LiteLlm("openai/gpt-4o") call lands as
    # ("litellm", "openai"), not as a Google/Gemini span, so backend
    # pricing + model lookups aren't skewed. TP-2128 round-3 PR #33.
    safe_set(span, GenAIAttributes.SYSTEM, gen_ai_system)
    safe_set(span, GenAIAttributes.REQUEST_MODEL, model)
    safe_set(span, GenAIAttributes.OPERATION_NAME, GenAIOperation.GENERATE_CONTENT)
    safe_set(span, GenAIAttributes.REQUEST_IS_STREAM, bool(stream))

    config = read(llm_request, "config")
    for cfg_key, agentic_key, gen_ai_key in (
        ("temperature", AgenticAttributes.REQUEST_TEMPERATURE, GenAIAttributes.REQUEST_TEMPERATURE),
        (
            "max_output_tokens",
            AgenticAttributes.REQUEST_MAX_TOKENS,
            GenAIAttributes.REQUEST_MAX_TOKENS,
        ),
        ("top_p", AgenticAttributes.REQUEST_TOP_P, GenAIAttributes.REQUEST_TOP_P),
        ("top_k", AgenticAttributes.REQUEST_TOP_K, GenAIAttributes.REQUEST_TOP_K),
    ):
        val = read(config, cfg_key) if config is not None else None
        if val is not None:
            safe_set(span, agentic_key, val)
            safe_set(span, gen_ai_key, val)

    system_instruction = read(config, "system_instruction") if config is not None else None
    if system_instruction:
        safe_set(span, AgenticAttributes.SYSTEM_INSTRUCTIONS, str(system_instruction))

    contents = read(llm_request, "contents")
    normalized = _normalize_contents(contents)
    if normalized:
        set_messages_if_capturing(span, input_messages=normalized)
        safe_set(span, GenAIAttributes.PROMPT, normalized)

    # ADK registered tool schemas:
    #   * config.tools carries google-genai FunctionDeclarations with
    #     model_dump(mode="json") giving a real schema. Prefer those.
    #   * tools_dict maps names -> BaseTool INSTANCES — json.dumps with
    #     default=str would yield "<FunctionTool object at 0x…>" and
    #     throw away the schema. Fall back to the tool NAMES instead so
    #     the span at least carries which tools were offered.
    tools_json: str | None = None
    cfg_tools = read(config, "tools") if config is not None else None
    if cfg_tools:
        serialized: list[Any] = []
        for t in cfg_tools:
            dump = getattr(t, "model_dump", None)
            if callable(dump):
                try:
                    serialized.append(dump(mode="json"))
                    continue
                except Exception:
                    pass
            serialized.append(str(t))
        try:
            tools_json = json.dumps(serialized, default=str)
        except (TypeError, ValueError):
            tools_json = None
    if tools_json is None:
        tools_dict = read(llm_request, "tools_dict")
        if tools_dict:
            try:
                names = list(tools_dict.keys()) if hasattr(tools_dict, "keys") else list(tools_dict)
                tools_json = json.dumps(names)
            except (TypeError, ValueError):
                tools_json = None
    if tools_json:
        safe_set(span, AgenticAttributes.REQUEST_TOOLS, tools_json)
        safe_set(span, GenAIAttributes.REQUEST_TOOLS, tools_json)


def _set_llm_response_attrs(span: DisseqtSpan, response: Any) -> None:
    resp_model = read(response, "model_version")
    # LlmResponse has no ``response_id``; ``interaction_id`` is the closest
    # correlator ADK exposes (populated for live sessions).
    resp_id = read(response, "interaction_id")
    safe_set(span, AgenticAttributes.RESPONSE_ID, resp_id)
    safe_set(span, AgenticAttributes.RESPONSE_MODEL, resp_model)
    safe_set(span, GenAIAttributes.RESPONSE_ID, resp_id)
    safe_set(span, GenAIAttributes.RESPONSE_MODEL, resp_model)

    finish_reason = read(response, "finish_reason")
    if finish_reason:
        safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, str(finish_reason))
        safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, [str(finish_reason)])

    error_code = read(response, "error_code")
    error_message = read(response, "error_message")
    if error_code:
        safe_set(span, AgenticAttributes.ERROR_CODE, str(error_code))
    if error_message:
        safe_set(span, AgenticAttributes.ERROR_MESSAGE, str(error_message))

    usage = read(response, "usage_metadata")
    if usage is not None:
        prompt_tokens = read(usage, "prompt_token_count") or 0
        output_tokens = read(usage, "candidates_token_count") or 0
        total = read(usage, "total_token_count") or (prompt_tokens + output_tokens)
        span.set_token_usage(prompt_tokens, output_tokens, total_tokens=total)
        safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, prompt_tokens)
        safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, output_tokens)
        safe_set(span, GenAIAttributes.USAGE_TOTAL_TOKENS, total)
        # Hidden-token breakdowns — same categories as raw Gemini.
        thoughts = read(usage, "thoughts_token_count")
        if thoughts:
            safe_set(span, AgenticAttributes.USAGE_THOUGHTS_TOKENS, thoughts)
        cached = read(usage, "cached_content_token_count")
        if cached:
            safe_set(span, AgenticAttributes.USAGE_CACHED_CONTENT_TOKENS, cached)

    content = read(response, "content")
    text = _extract_content_text(content)
    if text:
        msgs = [{"role": "model", "content": text}]
        set_messages_if_capturing(span, output_messages=msgs)
        safe_set(span, GenAIAttributes.COMPLETION, msgs)

    tool_calls = _extract_tool_calls(content, resp_id)
    if tool_calls:
        safe_set(span, AgenticAttributes.TOOL_CALLS, tool_calls)
        _notify_planned_tool_calls(tool_calls)
        safe_set(span, GenAIAttributes.TOOL_CALLS, tool_calls)
        set_first_tool_call_attrs(span, tool_calls)


# ---------------------------------------------------------------------
# Streaming accumulator for BaseLlm.generate_content_async
# ---------------------------------------------------------------------
class _LlmStreamState:
    """
    Aggregates ``LlmResponse`` chunks across a ``generate_content_async``
    async generator.

    ADK's contract: ``stream=True`` yields multiple ``partial=True``
    chunks (text deltas, no final usage) followed by exactly one
    ``partial=False`` aggregated chunk carrying the full ``content`` and
    the authoritative ``usage_metadata``. Non-streaming yields a single
    ``partial=False`` chunk. We forward every chunk unchanged to the
    caller (mustn't break their loop) and stamp attributes on the span
    only from the final ``partial=False`` chunk to avoid double-counting
    tokens or emitting a partial text buffer with wrong usage.
    """

    def __init__(self) -> None:
        self._final: Any = None
        # Fallback: if the stream ended without a partial=False chunk
        # (protocol violation, empty stream, cancelled), we still finalize
        # from the last chunk we saw so the span isn't left blank.
        self._last: Any = None

    def absorb(self, chunk: Any) -> None:
        self._last = chunk
        partial = read(chunk, "partial")
        # partial is None on non-streaming responses and False on the
        # final streaming chunk — both count as "the full response".
        if partial is not True:
            self._final = chunk

    def finalize(self, span: DisseqtSpan) -> None:
        response = self._final if self._final is not None else self._last
        if response is None:
            return
        _set_llm_response_attrs(span, response)


# ---------------------------------------------------------------------
# Runner.run_async wrapper
# ---------------------------------------------------------------------
def _extract_runner_metadata(instance: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull app_name, user_id, session_id, root-agent name from a Runner call."""
    meta: dict[str, Any] = {}
    app_name = read(instance, "app_name")
    if app_name:
        meta["app_name"] = app_name
    agent = read(instance, "agent")
    agent_name = read(agent, "name")
    if agent_name:
        meta["agent_name"] = agent_name
    user_id = kwargs.get("user_id")
    if user_id:
        meta["user_id"] = user_id
    session_id = kwargs.get("session_id")
    if session_id:
        meta["session_id"] = session_id
    return meta


def _set_runner_attrs(span: DisseqtSpan, meta: dict[str, Any]) -> None:
    span.set_operation(AgenticOperation.INVOKE_AGENT)
    # gen_ai.system was previously hardcoded to "gemini" on the AGENT_EXEC
    # span even for runs that routed through LiteLLM / Anthropic. The
    # runner span isn't an LLM call anyway, so dropping the tag beats
    # mislabeling it; the per-LLM MODEL_EXEC span derives its own
    # gen_ai.system via _detect_provider.
    safe_set(span, GenAIAttributes.OPERATION_NAME, AgenticOperation.INVOKE_AGENT)
    if meta.get("agent_name"):
        # set_agent_info stamps agentic.agent.name; also mirror app_name
        # since ADK apps often share the agent name with the app name.
        span.set_agent_info(agent_name=meta["agent_name"])
    if meta.get("app_name"):
        safe_set(span, "agentic.app.name", meta["app_name"])
    if meta.get("user_id"):
        safe_set(span, "agentic.session.user_id", str(meta["user_id"]))
    if meta.get("session_id"):
        safe_set(span, "agentic.session.id", str(meta["session_id"]))


def runner_run_async(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # NOTE: run_async is an async-gen function. Calling it returns
        # an async_generator object without executing the body — so
        # this wrapper is deliberately synchronous. Making it ``async
        # def`` would force callers to ``await runner.run_async(...)``
        # which breaks ADK's public contract (they iterate directly).
        meta = safe_extract(_extract_runner_metadata, instance, kwargs)
        scope = open_llm_span(instrumentor.client, "adk.runner.run_async", SpanKind.AGENT_EXEC)
        span = scope.span
        safe_call(_set_runner_attrs, span, meta or {})

        try:
            agen = wrapped(*args, **kwargs)
        except BaseException as exc:
            # No agg installed yet at this point; pass install_agg=False.
            _finalize_agent_scope(scope, span, agg_token=None, exc=exc)
            raise
        # Aggregator install is deferred to the first __anext__ so the
        # contextvars ``set`` and ``reset`` happen in the SAME asyncio
        # Context. Doing it here (in the sync wrapper) would create the
        # Token in the caller's context and later reset it from inside
        # the Task's copied context — the exact "Token created in a
        # different Context" leak flagged in adk-python#949.
        return _AgentGenScope(
            agen=agen,
            scope=scope,
            span=span,
            install_agg=True,
        )

    return wrapper


# ---------------------------------------------------------------------
# BaseAgent.run_async wrapper
# ---------------------------------------------------------------------
def _set_agent_attrs(span: DisseqtSpan, instance: Any, parent_context: Any) -> None:
    span.set_operation(AgenticOperation.INVOKE_AGENT)
    safe_set(span, GenAIAttributes.OPERATION_NAME, AgenticOperation.INVOKE_AGENT)
    name = read(instance, "name")
    if name:
        span.set_agent_info(agent_name=name)
    description = read(instance, "description")
    if description:
        safe_set(span, "agentic.agent.description", str(description))
    # parent_context (an InvocationContext) exposes invocation_id and branch.
    invocation_id = read(parent_context, "invocation_id")
    if invocation_id:
        safe_set(span, "agentic.invocation.id", str(invocation_id))
    branch = read(parent_context, "branch")
    if branch:
        safe_set(span, "agentic.invocation.branch", str(branch))


def agent_run_async(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        agent_name = read(instance, "name") or "agent"
        scope = open_llm_span(instrumentor.client, f"adk.agent.{agent_name}", SpanKind.AGENT_EXEC)
        span = scope.span
        parent_context = args[0] if args else kwargs.get("parent_context")
        safe_call(_set_agent_attrs, span, instance, parent_context)

        try:
            agen = wrapped(*args, **kwargs)
        except BaseException as exc:
            _finalize_agent_scope(scope, span, agg_token=None, exc=exc)
            raise
        # Deferred install (see runner_run_async note). BaseAgent frames
        # nest inside a Runner frame that already installed the aggregator;
        # _install_aggregator_if_absent() at __anext__ time will observe
        # that and skip, so nested agent turns don't fragment tool_calls
        # per turn. Direct-invocation (no Runner) still installs at
        # __anext__ time.
        return _AgentGenScope(
            agen=agen,
            scope=scope,
            span=span,
            install_agg=True,
        )

    return wrapper


# ---------------------------------------------------------------------
# BaseTool.run_async wrapper
# ---------------------------------------------------------------------
def _set_tool_request_attrs(
    span: DisseqtSpan, instance: Any, tool_args: Any, tool_context: Any
) -> None:
    span.set_operation(AgenticOperation.EXECUTE_TOOL)
    safe_set(span, GenAIAttributes.OPERATION_NAME, AgenticOperation.EXECUTE_TOOL)
    name = read(instance, "name")
    if name:
        span.set_tool_info(tool_name=name)
        safe_set(span, GenAIAttributes.TOOL_NAME, name)
    description = read(instance, "description")
    if description:
        safe_set(span, "agentic.tool.description", str(description))
    call_id = read(tool_context, "function_call_id")
    if call_id:
        safe_set(span, AgenticAttributes.TOOL_CALL_ID, str(call_id))
        safe_set(span, GenAIAttributes.TOOL_CALL_ID, str(call_id))
    if tool_args is not None:
        # Serialize now with default=str so a tool-arg containing a
        # non-JSON-native type (e.g. datetime, UUID, Path, bytes) can't
        # make the span's later json.dumps call blow up and drop the
        # whole TOOL_EXEC span. Mirrors what _extract_tool_calls does.
        serialized = _safe_json(tool_args)
        safe_set(span, AgenticAttributes.TOOL_ARGS, serialized)
        safe_set(span, GenAIAttributes.TOOL_ARGS, serialized)


def _set_tool_result_attrs(span: DisseqtSpan, result: Any) -> None:
    if result is None:
        return
    serialized = _safe_json(result)
    safe_set(span, AgenticAttributes.TOOL_RESULT, serialized)
    safe_set(span, GenAIAttributes.TOOL_RESULT, serialized)


def _safe_json(value: Any) -> str:
    """json.dumps with default=str so one odd type can't drop the span.

    The span buffer later runs ``json.dumps(self.attributes)`` with no
    ``default`` fallback; a raw tool arg or result containing a
    ``datetime`` / ``UUID`` / ``Path`` / ``bytes`` would otherwise crash
    serialization and the whole span is silently dropped. Pre-serializing
    here guarantees every value we hand off is a plain string.
    """
    try:
        if isinstance(value, (str, bytes)):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _record_tool_outcome(
    instance: Any,
    tool_context: Any,
    tool_args: Any,
    result: Any,
    exc: BaseException | None,
) -> None:
    """Merge into the current AGENT_EXEC aggregator so the tool-* validators see it."""
    agg = _current_agg.get()
    if agg is None or agg.closed:
        return
    call_id = read(tool_context, "function_call_id")
    if not call_id:
        return
    name = read(instance, "name")
    status = "success" if exc is None else _classify_tool_error(exc)
    agg.add_result(
        str(call_id),
        name=name,
        arguments=tool_args,
        result=None if exc is not None else result,
        status=status,
    )


def _classify_tool_error(exc: BaseException) -> str:
    # asyncio.TimeoutError / TimeoutError → timeout; anything else → error.
    # We could reach for the concurrent.futures.TimeoutError alias too, but
    # asyncio.TimeoutError IS TimeoutError in 3.11+.
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "error"


def tool_run_async(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        tool_args = kwargs.get("args")
        tool_context = kwargs.get("tool_context")
        tool_name = read(instance, "name") or "tool"
        scope = open_llm_span(instrumentor.client, f"adk.tool.{tool_name}", SpanKind.TOOL_EXEC)
        span = scope.span
        safe_call(_set_tool_request_attrs, span, instance, tool_args, tool_context)
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as exc:
            # Aggregator update BEFORE scope exit — the finally in
            # _AgentGenScope flushes aggregator onto the AGENT_EXEC span,
            # so this outcome needs to land before that flush happens.
            safe_call(_record_tool_outcome, instance, tool_context, tool_args, None, exc)
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        safe_call(_set_tool_result_attrs, span, result)
        safe_call(_record_tool_outcome, instance, tool_context, tool_args, result, None)
        scope.__exit__(None, None, None)
        return result

    return wrapper


# ---------------------------------------------------------------------
# BaseLlm.generate_content_async wrapper
# ---------------------------------------------------------------------
def llm_generate_content_async(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        llm_request = args[0] if args else kwargs.get("llm_request")
        stream = args[1] if len(args) > 1 else kwargs.get("stream", False)
        model = read(llm_request, "model") or read(instance, "model") or "adk.llm"
        scope = open_llm_span(instrumentor.client, f"adk.llm.{model}", SpanKind.MODEL_EXEC)
        span = scope.span
        safe_call(_set_llm_request_attrs, span, llm_request, bool(stream), instance)

        try:
            agen = wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        state = _LlmStreamState()
        return AsyncStreamWrapper(
            stream=agen,
            scope=scope,
            on_chunk=state.absorb,
            on_finish=lambda: state.finalize(span),
        )

    return wrapper


# ---------------------------------------------------------------------
# BaseMemoryService.search_memory wrapper
# ---------------------------------------------------------------------
def _set_memory_search_attrs(span: DisseqtSpan, instance: Any, kwargs: dict[str, Any]) -> None:
    span.set_operation("search_memory")
    # ADK's memory service is a first-class RAG source; tag the backend
    # so dashboards can distinguish InMemory / VertexAiRag / MemoryBank.
    backend = type(instance).__name__ if instance is not None else "memory"
    safe_set(span, "agentic.rag.backend", backend)
    query = kwargs.get("query")
    if query:
        safe_set(span, "agentic.rag.query", str(query))
    app_name = kwargs.get("app_name")
    if app_name:
        safe_set(span, "agentic.app.name", str(app_name))
    user_id = kwargs.get("user_id")
    if user_id:
        safe_set(span, "agentic.session.user_id", str(user_id))


def _set_memory_search_result_attrs(span: DisseqtSpan, result: Any) -> None:
    memories = read(result, "memories")
    if memories is None:
        return
    try:
        count = len(memories)
    except TypeError:
        return
    safe_set(span, "agentic.rag.result_count", count)


def memory_search_memory(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    """
    Wrap ``BaseMemoryService.search_memory`` as a RAG_EXEC span.

    All ADK memory services (``InMemoryMemoryService``,
    ``VertexAiRagMemoryService``, ``VertexAiMemoryBankService``) override
    the abstract ``search_memory(*, app_name, user_id, query)`` coroutine
    on ``BaseMemoryService``, so one patch covers the tree. Return value
    is a ``SearchMemoryResponse`` with ``.memories: list[MemoryEntry]``
    — we record the count as ``agentic.rag.result_count``.
    """

    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "adk.memory.search", SpanKind.RAG_EXEC)
        span = scope.span
        safe_call(_set_memory_search_attrs, span, instance, kwargs)
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        safe_call(_set_memory_search_result_attrs, span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


# ---------------------------------------------------------------------
# RemoteA2aAgent._run_async_impl wrapper (A2A CLIENT span)
# ---------------------------------------------------------------------
def _set_a2a_attrs(span: DisseqtSpan, instance: Any) -> None:
    # CLIENT-kind: this process is calling out to a remote agent over
    # HTTP. Sits INSIDE the BaseAgent.run_async AGENT_EXEC (which fires
    # first via inheritance) so the trace tree reads:
    #     adk.agent.<name>  AGENT_EXEC
    #     └── adk.a2a.send  CLIENT   ← this span
    span.set_operation("a2a.send")
    name = read(instance, "name")
    if name:
        safe_set(span, "agentic.a2a.from", str(name))
    # RemoteA2aAgent stores the target as a URL or agent card source.
    url = read(instance, "agent_card") or read(instance, "url")
    if url:
        safe_set(span, "agentic.a2a.to", str(url))
    timeout = read(instance, "_timeout")
    if timeout is not None:
        safe_set(span, "agentic.a2a.timeout_s", timeout)


def a2a_run_async_impl(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    """
    Wrap ``RemoteA2aAgent._run_async_impl`` as a CLIENT span around the
    A2A network hop.

    The outer ``BaseAgent.run_async`` patch already emits an AGENT_EXEC
    span for this agent — we deliberately choose ``_run_async_impl``
    (the async-gen that actually does ``_compat.send_message(...)`` at
    line ~1640 of google-adk 2.9.1) so the CLIENT span nests *inside*
    the AGENT_EXEC rather than replacing it. This mirrors OTel's
    convention: AGENT_EXEC = "run this agent turn", CLIENT = "send
    request to remote peer, wait for reply".

    Reuses ``_AgentGenScope`` with ``install_agg=False`` since:
      * we don't want a tool-call aggregator on this scope (aggregation
        belongs on the enclosing AGENT_EXEC scope);
      * we DO want the same idempotent-close + aclose-forwarding
        guarantees that keep async-gen wrappers deadlock-safe.
    """

    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "adk.a2a.send", SpanKind.CLIENT)
        span = scope.span
        safe_call(_set_a2a_attrs, span, instance)

        try:
            agen = wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return _AgentGenScope(
            agen=agen,
            scope=scope,
            span=span,
            install_agg=False,
        )

    return wrapper


# ---------------------------------------------------------------------
# AGENT_EXEC async-generator scope
# ---------------------------------------------------------------------
async def _AgentGenScope(  # noqa: N802 — kept PascalCase for API compat
    agen: Any,
    scope: _SpanScope,
    span: DisseqtSpan,
    install_agg: bool,
) -> Any:
    """
    Native async-generator wrapper around the ADK runner / agent /
    A2A async generators.

    PREVIOUSLY a plain class with ``__anext__`` / ``__aexit__``, which
    looked right but didn't play well with ``async for ... break``: the
    event loop's asyncgen finalizer only runs for NATIVE async
    generators, so a plain-class wrapper that got abandoned on break
    had its finalize path skipped — the AGENT_EXEC root span never
    emitted, the tool-call aggregator stayed set in the ContextVar,
    and the next run nested into the stale trace (ADK's quickstart
    does exactly ``async for ev in agen: if is_final_response(): break``).

    Rewriting as a real ``async def ... yield`` function means CPython's
    asyncgen hooks guarantee ``finally`` runs on:
      * normal exhaustion (StopAsyncIteration)
      * caller ``break`` / ``return`` inside ``async for``
      * caller ``aclose()`` or explicit cancellation
      * GC / event-loop shutdown via ``set_asyncgen_hooks``

    Deferred aggregator install:
      ``_install_aggregator_if_absent()`` is called INSIDE this
      generator's own body, which runs inside the task's own copied
      Context — avoids the adk-python#949 "Token created in a different
      Context" leak that would happen if the install ran in the
      caller's context and the reset ran in the task's.
    """
    agg_token: Any = _install_aggregator_if_absent() if install_agg else None
    exc_seen: BaseException | None = None
    try:
        async for item in agen:
            yield item
    except BaseException as exc:
        # CancelledError / GeneratorExit / KeyboardInterrupt land here.
        exc_seen = exc
        raise
    finally:
        # Forward aclose to the inner generator so ADK's own cleanup
        # (contextvar reset, invocation-scope teardown) still runs on
        # every exit path.
        inner_aclose = getattr(agen, "aclose", None)
        if callable(inner_aclose):
            try:
                await inner_aclose()
            except BaseException:  # noqa: BLE001 — never propagate cleanup failure
                pass
        _finalize_agent_scope(scope, span, agg_token=agg_token, exc=exc_seen)


def _finalize_agent_scope(
    scope: _SpanScope,
    span: DisseqtSpan,
    *,
    agg_token: Any,
    exc: BaseException | None,
) -> None:
    """
    Flush the tool-call aggregator (if this scope installed one) onto
    the AGENT_EXEC span, reset the contextvar, and close the span
    scope. Order matters — aggregator flush must run BEFORE ``scope.
    __exit__`` so the merged tool_calls ship with the span.
    """
    if agg_token is not None:
        agg = _current_agg.get()
        if agg is not None:
            try:
                agg.flush_onto(span)
            except Exception:  # noqa: BLE001 — never break the caller
                pass
        try:
            _current_agg.reset(agg_token)
        except (LookupError, ValueError):
            # Token was created in a different context — a known ADK
            # gotcha (google/adk-python#949). Nothing sensible to do
            # here; leaving the var alone is safer than raising.
            pass
    if exc is not None:
        scope.__exit__(type(exc), exc, exc.__traceback__)
    else:
        scope.__exit__(None, None, None)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _install_aggregator_if_absent() -> Any:
    """
    Install a fresh ``_ToolCallAggregator`` into the current context IFF
    none is already active. Returns the contextvars Token to reset with
    later (None if we didn't install).
    """
    if _current_agg.get() is not None:
        return None
    return _current_agg.set(_ToolCallAggregator())


def safe_extract(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Invoke ``fn`` and return its result, or an empty dict on any failure.
    Metadata extraction must never break the wrapped call.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return {}
