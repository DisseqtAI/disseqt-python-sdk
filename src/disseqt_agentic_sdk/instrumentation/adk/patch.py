"""
Google ADK patch functions.

How every wrapper here handles context
--------------------------------------
ADK is asyncio-first, and several runs routinely share one event loop
(``asyncio.gather`` over ``runner.run_async(...)`` calls). Thread-local
"current trace / span" cannot tell those runs apart, so these wrappers
never read or write it while ADK code is running:

  * ``_capture_parent()`` snapshots the trace and span that are current
    when the wrapper is entered. ``_open_adk_span()`` parents the new
    span off that snapshot explicitly.
  * Around every resumption of the wrapped coroutine / async generator
    the wrapper enters an *isolated context* (a ``ContextVar`` overlay,
    see ``disseqt_agentic_sdk.context``) seeded with its own trace and
    span. Nested ADK wrappers — and any other instrumentation running
    underneath, including in child tasks and ``copy_context()`` worker
    threads — therefore see the right parent, per task.
  * The overlay, the tool-call aggregator and the provider-span
    suppression flag are all set and reset inside a single resumption,
    never across a ``yield``. Nothing leaks into the caller between two
    events, and no ``Token`` is ever reset from a different ``Context``
    (the adk-python#949 failure mode).

Async-generator wrappers are ``_AdkGenScope`` objects rather than native
async generators so that an abandoned run is finalized when its last
reference is dropped — see that class for the ``async for ... break``
story.

Other invariants:

  * Async-gen wrappers call the wrapped function synchronously (async-gen
    functions return a generator object without running the body); no
    await is held across a lock.
  * All request/response extraction goes through ``safe_call`` / ``read``
    which never propagate exceptions into the caller.
  * No ``asyncio.run`` / ``loop.run_until_complete`` — those would
    deadlock inside an already-running event loop (a known ADK footgun,
    see google/adk-python#755).
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import json
import weakref
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.context import (
    enter_isolated_context,
    exit_isolated_context,
    get_current_span,
    get_current_trace,
)
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._tool_result import (
    _current_agg,
    _notify_planned_tool_calls,
    _ToolCallAggregator,
)
from disseqt_agentic_sdk.instrumentation._utils import (
    _get_or_bootstrap_trace,
    _SpanScope,
    read,
    safe_call,
    safe_set,
    set_first_tool_call_attrs,
    set_messages_if_capturing,
    suppress_provider_spans,
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
    from disseqt_agentic_sdk.client import DisseqtAgenticClient
    from disseqt_agentic_sdk.instrumentation.adk.instrumentor import AdkInstrumentor
    from disseqt_agentic_sdk.span import DisseqtSpan
    from disseqt_agentic_sdk.trace import DisseqtTrace


PROVIDER = AgenticProvider.GOOGLE

# The tool / memory service whose call is currently being traced in this
# context. ADK reaches a tool through a call site (``_call_tool_async``)
# that we wrap, and the tool's own ``run_async`` is wrapped too as a
# backstop for direct calls and other ADK layouts; ``super().run_async()``
# chains add further layers. Whichever wrapper is outermost opens the
# span and records the instance here, and every wrapper underneath that
# sees the *same instance* passes straight through — one call, one span.
_active_tool: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "disseqt_adk_active_tool", default=None
)
_active_memory: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "disseqt_adk_active_memory", default=None
)


# ---------------------------------------------------------------------
# Parent capture + span lifecycle
# ---------------------------------------------------------------------
def _capture_parent() -> tuple[DisseqtTrace | None, DisseqtSpan | None]:
    """
    Snapshot the trace and span that are current *right now*.

    Called at wrapper entry. Inside an ADK run this reads the isolated
    context of the enclosing wrapper (per task); at the top of a run it
    reads whatever trace the user opened around the call, if any. An
    ended trace is ignored — starting a span on it would raise — and so
    is a span that belongs to some other trace.
    """
    trace = get_current_trace()
    if trace is None or trace.is_ended:
        return None, None
    span = get_current_span()
    if span is not None and str(span.trace_id) != str(trace.trace_id):
        span = None
    return trace, span


def _open_adk_span(client: DisseqtAgenticClient, name: str, kind: SpanKind | str) -> _SpanScope:
    """
    Open a span parented off the snapshot taken by ``_capture_parent``.

    With no parent trace a fresh one is bootstrapped (and owned by the
    returned scope). The whole operation runs inside a throwaway isolated
    context so that neither the new trace nor the new span is published
    to thread-local storage, where a concurrent run would pick it up.
    """
    parent_trace, parent_span = _capture_parent()
    token = enter_isolated_context(parent_trace, parent_span)
    try:
        trace, owns_trace = _get_or_bootstrap_trace(client, name)
        span = trace.start_span(
            name,
            kind,
            parent_span_id=parent_span.span_id if parent_span is not None else None,
        )
        return _SpanScope(span=span, trace=trace, owns_trace=owns_trace)
    finally:
        exit_isolated_context(token)


def _close_adk_span(scope: _SpanScope, exc: BaseException | None) -> None:
    """
    End the span (and its trace, if the scope owns it).

    ``GeneratorExit`` is not an error: it is what Python injects when a
    consumer stops iterating early (``break`` then ``aclose()``), which
    is how ADK's own quickstart ends a run. Runs inside a throwaway
    isolated context for the same reason as ``_open_adk_span``.
    """
    if isinstance(exc, GeneratorExit):
        exc = None
    token = enter_isolated_context(scope.trace, scope.span)
    try:
        if exc is None:
            scope.__exit__(None, None, None)
        else:
            scope.__exit__(type(exc), exc, exc.__traceback__)
    finally:
        exit_isolated_context(token)


class _AdkGenScope:
    """
    Async-iterator wrapper that ties a span to an ADK async generator
    (``Runner.run_async``, ``BaseAgent.run_async``, the model response
    stream, the A2A hop).

    Finalization — span closed, owned aggregator flushed, owned trace
    ended — happens exactly once, on whichever comes first:

      * the inner generator is exhausted or raises;
      * ``aclose()`` is called (ADK wraps its own generators in
        ``Aclosing``; ``GeneratorExit`` is recorded as a clean exit);
      * the wrapper is garbage collected (``__del__``).

    The last case is what makes ``async for event in runner.run_async():
    ... break`` work. Nothing tells an iterator that its consumer left
    the loop; a native async generator only finds out when the event loop
    later schedules its ``aclose()``, or at loop shutdown. A plain object
    is told synchronously, the moment the loop statement drops the only
    reference. A caller that keeps its own reference
    (``agen = runner.run_async(...)``) and breaks without closing keeps
    the span open until that reference goes away or ``aclose()`` is
    called — the same lifetime the underlying ADK generator has.

    Per-resumption state (isolated context, aggregator, suppression) is
    installed by ``_step()`` and is never held across a ``yield``, so a
    ``break`` leaves nothing behind in the caller's context either.
    """

    def __init__(
        self,
        agen: Any,
        scope: _SpanScope,
        *,
        install_agg: bool = False,
        suppress_provider_spans: bool = False,
        on_item: Callable[[Any], None] | None = None,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        self._finalized = False
        self._agen = agen
        self._scope = scope
        self._install_agg = install_agg
        self._suppress_provider_spans = suppress_provider_spans
        self._on_item = on_item
        self._on_finish = on_finish
        # Created on the first resumption, and only if no enclosing scope
        # already aggregates: a BaseAgent frame nested in a Runner frame
        # must feed the Runner's aggregator, not fragment tool calls.
        self._agg: _ToolCallAggregator | None = None
        self._agg_decided = not install_agg

    @contextlib.contextmanager
    def _step(self) -> Iterator[None]:
        if not self._agg_decided:
            self._agg_decided = True
            if _current_agg.get() is None:
                self._agg = _ToolCallAggregator()
        ctx_token = enter_isolated_context(self._scope.trace, self._scope.span)
        # A generator boundary starts a fresh call chain: a tool invoked
        # beneath it is a new call even if it is the same tool instance
        # (an AgentTool whose agent ends up calling it again).
        resets: list[tuple[contextvars.ContextVar[Any], contextvars.Token[Any]]] = [
            (_active_tool, _active_tool.set(None)),
            (_active_memory, _active_memory.set(None)),
        ]
        if self._agg is not None:
            resets.append((_current_agg, _current_agg.set(self._agg)))
        try:
            if self._suppress_provider_spans:
                # Scoped to this resumption of this model call, in this
                # task only: a task started meanwhile copies the context
                # but is not suppressed (see suppress_provider_spans).
                with suppress_provider_spans():
                    yield
            else:
                yield
        finally:
            for var, token in reversed(resets):
                with contextlib.suppress(ValueError, RuntimeError):
                    var.reset(token)
            exit_isolated_context(ctx_token)

    def __aiter__(self) -> _AdkGenScope:
        return self

    async def __anext__(self) -> Any:
        if self._finalized:
            raise StopAsyncIteration
        try:
            with self._step():
                item = await self._agen.__anext__()
        except StopAsyncIteration:
            self._finalize(None)
            raise
        except BaseException as exc:
            # BaseException so CancelledError (client disconnect) and
            # KeyboardInterrupt still close the span.
            self._finalize(exc)
            raise
        if self._on_item is not None:
            with contextlib.suppress(Exception):
                self._on_item(item)
        return item

    async def aclose(self) -> None:
        """Close the inner generator, then finalize. Idempotent."""
        if self._finalized:
            return
        inner_aclose = getattr(self._agen, "aclose", None)
        try:
            if callable(inner_aclose):
                # Inside a step so ADK's own teardown still sees this
                # run's context. Cleanup errors must not mask the reason
                # the consumer is closing; cancellation still propagates.
                with self._step(), contextlib.suppress(Exception):
                    await inner_aclose()
        finally:
            self._finalize(None)

    def _finalize(self, exc: BaseException | None) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._on_finish is not None:
            with contextlib.suppress(Exception):
                self._on_finish()
        if self._agg is not None:
            # Before the span closes, so the fused tool_calls ship with it.
            with contextlib.suppress(Exception):
                self._agg.flush_onto(self._scope.span)
        with contextlib.suppress(Exception):
            _close_adk_span(self._scope, exc)

    def __del__(self) -> None:
        # getattr: __init__ may not have run to completion.
        if getattr(self, "_finalized", True):
            return
        with contextlib.suppress(BaseException):
            self._finalize(None)


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


# Planned calls whose id this module made up, and executed calls already
# paired with one, per aggregator. See ``_pair_planned_with_executed``.
_made_up_ids: weakref.WeakKeyDictionary[_ToolCallAggregator, set[str]] = weakref.WeakKeyDictionary()
_paired_ids: weakref.WeakKeyDictionary[_ToolCallAggregator, set[str]] = weakref.WeakKeyDictionary()


def _pair_planned_with_executed(content: Any, tool_calls: list[dict[str, Any]]) -> None:
    """
    Give a planned call that came without an id the id its execution used.

    A model that sends no function-call id gets one from ADK
    (``adk-<uuid>``) when the response is turned into an event, and the
    tool then runs under that id. ADK 2.x writes the id onto the response
    object this module is holding, so plan and execution agree. ADK 1.x
    writes it onto a copy, so the plan is left with the placeholder from
    ``_extract_tool_calls`` and the same call would be listed twice on
    the AGENT_EXEC span: once planned, once executed.

    Whichever side arrives second adopts the other: here when the tool
    already ran, in ``_record_tool_outcome`` when it runs later. Calls
    are paired by tool name, in order.
    """
    agg = _current_agg.get()
    if agg is None or agg.closed:
        return
    parts = [
        p
        for p in (read(content, "parts") or [])
        if isinstance(read(read(p, "function_call"), "name"), str)
        and read(read(p, "function_call"), "name")
    ]
    if len(parts) != len(tool_calls):
        return
    made_up = _made_up_ids.setdefault(agg, set())
    paired = _paired_ids.setdefault(agg, set())
    real_ids = {tc["id"] for part, tc in zip(parts, tool_calls, strict=True) if _has_id(part)}
    for part, tc in zip(parts, tool_calls, strict=True):
        if _has_id(part):
            continue
        for call_id, entry in agg._calls.items():
            if (
                "status" in entry
                and entry.get("name") == tc["name"]
                and call_id not in paired
                and call_id not in real_ids
                and call_id not in made_up
            ):
                tc["id"] = call_id
                paired.add(call_id)
                break
        else:
            made_up.add(tc["id"])


def _has_id(part: Any) -> bool:
    return bool(read(read(part, "function_call"), "id"))


def _adopt_planned_entry(agg: _ToolCallAggregator, call_id: str, name: Any) -> None:
    """
    The other direction of ``_pair_planned_with_executed``: a tool is
    about to be recorded under ``call_id``, which no plan mentions. Move
    a planned entry with a made-up id and the same tool name under it.
    """
    if call_id in agg._calls:
        return
    made_up = _made_up_ids.get(agg)
    if not made_up:
        return
    for placeholder in list(made_up):
        entry = agg._calls.get(placeholder)
        if entry is None or "status" in entry or entry.get("name") != name:
            continue
        del agg._calls[placeholder]
        made_up.discard(placeholder)
        entry["id"] = call_id
        agg._calls[call_id] = entry
        _paired_ids.setdefault(agg, set()).add(call_id)
        return


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
        safe_call(_pair_planned_with_executed, content, tool_calls)
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
        scope = _open_adk_span(instrumentor.client, "adk.runner.run_async", SpanKind.AGENT_EXEC)
        safe_call(_set_runner_attrs, scope.span, meta or {})

        try:
            agen = wrapped(*args, **kwargs)
        except BaseException as exc:
            _close_adk_span(scope, exc)
            raise
        return _AdkGenScope(agen, scope, install_agg=True)

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
        scope = _open_adk_span(instrumentor.client, f"adk.agent.{agent_name}", SpanKind.AGENT_EXEC)
        parent_context = args[0] if args else kwargs.get("parent_context")
        safe_call(_set_agent_attrs, scope.span, instance, parent_context)

        try:
            agen = wrapped(*args, **kwargs)
        except BaseException as exc:
            _close_adk_span(scope, exc)
            raise
        # install_agg=True, but the aggregator is only created if none is
        # active on the first resumption: nested under a Runner frame the
        # Runner's aggregator is reused; a directly-invoked agent (no
        # Runner) gets its own.
        return _AdkGenScope(agen, scope, install_agg=True)

    return wrapper


# ---------------------------------------------------------------------
# Tool wrappers (TOOL_EXEC)
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
    _adopt_planned_entry(agg, str(call_id), name)
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


async def _trace_tool_call(
    instrumentor: AdkInstrumentor,
    tool: Any,
    tool_args: Any,
    tool_context: Any,
    call: Callable[[], Any],
) -> Any:
    """Run ``await call()`` inside a TOOL_EXEC span for ``tool``."""
    tool_name = read(tool, "name") or "tool"
    scope = _open_adk_span(instrumentor.client, f"adk.tool.{tool_name}", SpanKind.TOOL_EXEC)
    span = scope.span
    safe_call(_set_tool_request_attrs, span, tool, tool_args, tool_context)

    ctx_token = enter_isolated_context(scope.trace, span)
    active_token = _active_tool.set(tool)
    result: Any = None
    exc_seen: BaseException | None = None
    try:
        result = await call()
    except BaseException as exc:
        # BaseException so cancellation still closes the span.
        exc_seen = exc
        raise
    finally:
        with contextlib.suppress(ValueError, RuntimeError):
            _active_tool.reset(active_token)
        exit_isolated_context(ctx_token)
        # A streaming tool hands back an async generator; there is no
        # result to serialize at this point.
        if exc_seen is None and not inspect.isasyncgen(result):
            safe_call(_set_tool_result_attrs, span, result)
        # Aggregator update BEFORE the span closes — the enclosing
        # AGENT_EXEC scope flushes the aggregator when it finalizes, so
        # this outcome has to land first.
        safe_call(
            _record_tool_outcome,
            tool,
            tool_context,
            tool_args,
            None if exc_seen is not None else result,
            exc_seen,
        )
        _close_adk_span(scope, exc_seen)
    return result


def tool_call_site(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    """
    Wrap one of ADK's tool call sites, ``_call_tool_async(tool, args,
    tool_context)`` or ``_call_tool_in_thread_pool(...)``.

    Every tool the LLM flow invokes goes through one of these, whatever
    its class. Where they live depends on the ADK release
    (``flows/llm_flows/functions.py`` in 1.x, ``_tool_caller.py`` in
    2.9, ``tools/_caller.py`` and ``tools/_thread_pool.py`` in 2.14), so
    the instrumentor looks them up by name rather than by module path —
    see ``AdkInstrumentor._wrap_tool_call_sites``.

    Passes through when an outer wrapper is already tracing this same
    tool instance (a re-exported binding wrapped twice, say).
    """

    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        tool = args[0] if args else kwargs.get("tool")
        if tool is not None and _active_tool.get() is tool:
            return await wrapped(*args, **kwargs)
        tool_args = args[1] if len(args) > 1 else kwargs.get("args")
        tool_context = args[2] if len(args) > 2 else kwargs.get("tool_context")
        return await _trace_tool_call(
            instrumentor, tool, tool_args, tool_context, lambda: wrapped(*args, **kwargs)
        )

    return wrapper


def tool_run_async(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    """
    Wrap ``<Tool>.run_async(*, args, tool_context)`` on a tool class.

    Installed on ``BaseTool`` and on every subclass that overrides
    ``run_async`` (see ``AdkInstrumentor._wrap_tool_classes``). It covers
    a tool run directly, outside the LLM flow, and it is what keeps tools
    traced on an ADK release whose call sites have moved somewhere this
    SDK does not know yet. Passes through when an outer wrapper is
    already tracing this same tool instance.
    """

    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if instance is not None and _active_tool.get() is instance:
            return await wrapped(*args, **kwargs)
        return await _trace_tool_call(
            instrumentor,
            instance,
            kwargs.get("args"),
            kwargs.get("tool_context"),
            lambda: wrapped(*args, **kwargs),
        )

    return wrapper


# ---------------------------------------------------------------------
# Model wrappers (MODEL_EXEC)
# ---------------------------------------------------------------------
def _trace_llm_stream(
    instrumentor: AdkInstrumentor,
    llm: Any,
    llm_request: Any,
    stream: bool,
    call: Callable[[], Any],
) -> Any:
    """
    Wrap the ``LlmResponse`` async generator returned by ``call()`` in a
    MODEL_EXEC span.

    While the model call is in flight, provider-SDK instrumentation
    underneath (google-genai for ``Gemini``, litellm for ``LiteLlm``, …)
    is suppressed: ``instrument_all`` enables both layers, and without
    this one turn would be reported — and its tokens counted — twice.
    """
    model = read(llm_request, "model") or read(llm, "model") or "adk.llm"
    scope = _open_adk_span(instrumentor.client, f"adk.llm.{model}", SpanKind.MODEL_EXEC)
    span = scope.span
    safe_call(_set_llm_request_attrs, span, llm_request, bool(stream), llm)

    try:
        agen = call()
    except BaseException as exc:
        _close_adk_span(scope, exc)
        raise
    state = _LlmStreamState()
    return _AdkGenScope(
        agen,
        scope,
        suppress_provider_spans=True,
        on_item=state.absorb,
        on_finish=lambda: state.finalize(span),
    )


def _resolve_flow_llm(invocation_context: Any) -> Any:
    """The ``BaseLlm`` the current agent is configured with, or None."""
    agent = read(invocation_context, "agent")
    if agent is None:
        return None
    try:
        # A property on LlmAgent; resolving it can raise for a
        # misconfigured agent, which the real call will report itself.
        return getattr(agent, "canonical_model", None)
    except Exception:  # noqa: BLE001
        return None


def _flow_is_streaming(invocation_context: Any) -> bool:
    mode = read(read(invocation_context, "run_config"), "streaming_mode")
    return getattr(mode, "name", None) == "SSE"


def llm_call_site(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    """
    Wrap ``BaseLlmFlow._run_and_handle_error(response_generator,
    invocation_context, llm_request, model_response_event, ...)``.

    This is the generator the LLM flow builds around
    ``llm.generate_content_async(...)`` for every model turn, so it sees
    the turn for any ``BaseLlm`` — including the ones that override
    ``generate_content_async`` without being on a patch list (``Gemma``,
    ``ApigeeLlm``, the OpenAI / OCI integrations, user subclasses). It is
    only reached when the model is really called: a ``before_model``
    callback that answers the turn short-circuits before it.
    """

    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        invocation_context = args[1] if len(args) > 1 else kwargs.get("invocation_context")
        llm_request = args[2] if len(args) > 2 else kwargs.get("llm_request")
        return _trace_llm_stream(
            instrumentor,
            _resolve_flow_llm(invocation_context),
            llm_request,
            _flow_is_streaming(invocation_context),
            lambda: wrapped(*args, **kwargs),
        )

    return wrapper


def llm_generate_content_async(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    """
    Wrap ``<Llm>.generate_content_async(llm_request, stream=False)`` on a
    model class. Only installed when the ``llm_call_site`` target does
    not exist in the installed ADK — the two would otherwise both open a
    span for the same turn.
    """

    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        llm_request = args[0] if args else kwargs.get("llm_request")
        stream = args[1] if len(args) > 1 else kwargs.get("stream", False)
        return _trace_llm_stream(
            instrumentor, instance, llm_request, bool(stream), lambda: wrapped(*args, **kwargs)
        )

    return wrapper


# ---------------------------------------------------------------------
# Memory wrappers (RAG_EXEC)
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


async def _trace_memory_search(
    instrumentor: AdkInstrumentor,
    service: Any,
    search_kwargs: dict[str, Any],
    call: Callable[[], Any],
) -> Any:
    """Run ``await call()`` inside a RAG_EXEC span for ``service``."""
    scope = _open_adk_span(instrumentor.client, "adk.memory.search", SpanKind.RAG_EXEC)
    span = scope.span
    safe_call(_set_memory_search_attrs, span, service, search_kwargs)

    ctx_token = enter_isolated_context(scope.trace, span)
    active_token = _active_memory.set(service)
    result: Any = None
    exc_seen: BaseException | None = None
    try:
        result = await call()
    except BaseException as exc:
        exc_seen = exc
        raise
    finally:
        with contextlib.suppress(ValueError, RuntimeError):
            _active_memory.reset(active_token)
        exit_isolated_context(ctx_token)
        if exc_seen is None:
            safe_call(_set_memory_search_result_attrs, span, result)
        _close_adk_span(scope, exc_seen)
    return result


def memory_call_site(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    """
    Wrap ``Context.search_memory(query)`` (``google.adk.agents.context``;
    ``ToolContext`` is an alias of it).

    This is the one place ADK calls ``memory_service.search_memory(...)``
    — ``load_memory`` / ``preload_memory`` and user tools all come
    through it — so it covers every memory backend, including the ones
    that override ``search_memory`` without calling ``super()``
    (``FirestoreMemoryService``, user services).
    """

    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        invocation_context = getattr(instance, "_invocation_context", None)
        service = read(invocation_context, "memory_service")
        if service is None:
            # ADK raises "Memory service is not available"; nothing was
            # searched, so there is no RAG span to record.
            return await wrapped(*args, **kwargs)
        search_kwargs = {
            "query": args[0] if args else kwargs.get("query"),
            "app_name": read(invocation_context, "app_name"),
            "user_id": read(invocation_context, "user_id"),
        }
        return await _trace_memory_search(
            instrumentor, service, search_kwargs, lambda: wrapped(*args, **kwargs)
        )

    return wrapper


def memory_search_memory(instrumentor: AdkInstrumentor) -> Callable[..., Any]:
    """
    Wrap ``<MemoryService>.search_memory(*, app_name, user_id, query)``
    on a memory-service class.

    Backstop for a service searched directly and for ADK releases without
    the ``Context.search_memory`` call site. Passes through when an outer
    wrapper is already tracing this same service instance. The response
    is a ``SearchMemoryResponse``; its ``.memories`` length is recorded
    as ``agentic.rag.result_count``.
    """

    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if instance is not None and _active_memory.get() is instance:
            return await wrapped(*args, **kwargs)
        return await _trace_memory_search(
            instrumentor, instance, kwargs, lambda: wrapped(*args, **kwargs)
        )

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

    ``install_agg=False``: tool-call aggregation belongs on the enclosing
    AGENT_EXEC scope, not on the transport span.
    """

    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = _open_adk_span(instrumentor.client, "adk.a2a.send", SpanKind.CLIENT)
        safe_call(_set_a2a_attrs, scope.span, instance)

        try:
            agen = wrapped(*args, **kwargs)
        except BaseException as exc:
            _close_adk_span(scope, exc)
            raise
        return _AdkGenScope(agen, scope, install_agg=False)

    return wrapper


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_extract(fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Invoke ``fn`` and return its result, or an empty dict on any failure.
    Metadata extraction must never break the wrapped call.
    """
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return {}
