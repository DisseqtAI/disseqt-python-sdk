"""The five phase-4a kinds: decision, clarification_relevance, semantic_arg, intent,
refusal_alternative. Each is only data and small functions; the engine never switches on a name."""

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from ...assertions.canonical import canonical_json
from ...models.matchers import parse_iso
from ..render import TEXT_CAP, as_text, fmt_conversation, fmt_list, fmt_tool_calls, truncate
from .base import JudgeKindSpec, Precheck, register_kind

_SPACES = re.compile(r"\s+")


def _get(inputs: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return inputs.get(key, default)


# ------------------------------------------------------------------------------------ decision
def _decision_blocks(i: Mapping[str, Any], order: int, sample: int) -> list[tuple[str, str]]:
    tools = _get(i, "available_tools")
    return [
        ("AVAILABLE_TOOLS", fmt_list(tools) if tools else "(not provided)"),
        ("CONVERSATION", fmt_conversation(i["conversation"])),
        ("TOOL_CALLS", fmt_tool_calls(i["tool_calls"])),
        ("AGENT_TURN", truncate(as_text(i["agent_text"]), TEXT_CAP)),
    ]


def _decision_post(parsed: Mapping[str, Any], i: Mapping[str, Any]) -> tuple[str, str] | None:
    if parsed.get("label") == "call" and not i.get("tool_calls"):
        return "invalid_verdict", "label 'call' but <TOOL_CALLS> is empty"
    return None


# ------------------------------------------------------------------------------------ clarification
def _clarify_blocks(i: Mapping[str, Any], order: int, sample: int) -> list[tuple[str, str]]:
    return [
        ("TOOL_SCHEMAS", as_text(_get(i, "tool_schemas", "(not provided)"))),
        ("MISSING_HINTS", fmt_list(_get(i, "missing_hints"))),
        ("CONVERSATION", fmt_conversation(i["conversation"])),
        ("AGENT_TURN", truncate(as_text(i["agent_text"]), TEXT_CAP)),
    ]


# ------------------------------------------------------------------------------------ semantic_arg
def _norm_text(text: str) -> str:
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    return None


def _date_only(text: str) -> bool:
    return "T" not in text and " " not in text and len(text) == 10


def _semantic_precheck(i: Mapping[str, Any]) -> Precheck | None:
    """Only ever answers ``equivalent`` (never short-circuits to ``not_equivalent``). An author's
    rubric may redefine equivalence, so any non-empty rubric skips the precheck entirely."""
    if str(_get(i, "rubric") or "").strip():
        return None
    a, b = i["reference"], i["actual"]
    try:
        if canonical_json(a) == canonical_json(b):
            return Precheck("equivalent", "identical values", "canonical_json_equal")
    except (TypeError, ValueError):
        pass
    if isinstance(a, str) and isinstance(b, str) and _norm_text(a) == _norm_text(b):
        return Precheck("equivalent", "equal after normalization", "normalized_string_equal")
    na, nb = _as_number(a), _as_number(b)  # real numbers only: "02134" is not 2134
    if na is not None and nb is not None and na == nb:
        return Precheck("equivalent", "equal numbers", "numbers_equal")
    if isinstance(a, str) and isinstance(b, str):
        try:
            da, db = parse_iso(a.strip()), parse_iso(b.strip())
        except ValueError:
            return None
        if _date_only(a.strip()) != _date_only(b.strip()):
            return None  # a date and a datetime are not the same thing
        if _date_only(a.strip()):
            same = da.date() == db.date()
        elif (da.tzinfo is None) == (db.tzinfo is None):
            same = da == db
        else:
            same = False
        if same:
            return Precheck("equivalent", "same date or instant", "iso_equal")
    return None


def _semantic_blocks(i: Mapping[str, Any], order: int, sample: int) -> list[tuple[str, str]]:
    first, second = (i["reference"], i["actual"]) if order == 0 else (i["actual"], i["reference"])
    tool = f"{i['tool']} argument {as_text(i['path'])}"
    if _get(i, "tool_description"):
        tool += f"\n{truncate(as_text(i['tool_description']), TEXT_CAP)}"
    if _get(i, "param_schema"):
        tool += f"\nparameter schema: {as_text(i['param_schema'])}"
    convo = _get(i, "conversation") or _get(i, "user_input") or "(not provided)"
    return [
        ("TOOL", tool),
        ("ARG_RUBRIC", as_text(_get(i, "rubric", "(none)")) or "(none)"),
        ("CONVERSATION", fmt_conversation(convo) if not isinstance(convo, str) else convo),
        ("VALUE_A", truncate(as_text(first), TEXT_CAP)),
        ("VALUE_B", truncate(as_text(second), TEXT_CAP)),
    ]


# ------------------------------------------------------------------------------------ intent
def _intent_labels(i: Mapping[str, Any]) -> tuple[str, ...]:
    labels = tuple(str(x) for x in i["labels"])
    return (*(x for x in labels if x not in ("other", "unsure")), "other", "unsure")


def _intent_blocks(i: Mapping[str, Any], order: int, sample: int) -> list[tuple[str, str]]:
    real = [x for x in _intent_labels(i) if x not in ("other", "unsure")]
    k = sample % len(real) if real else 0
    rotated = [*real[k:], *real[:k], "other", "unsure"]  # order carries no meaning: rotate it
    return [
        ("LABELS", "\n".join(f"- {x}" for x in rotated)),
        ("CONVERSATION", fmt_conversation(i["conversation"])),
        ("TURN", str(i["turn"])),
    ]


# ------------------------------------------------------------------------------------ refusal
def _refusal_blocks(i: Mapping[str, Any], order: int, sample: int) -> list[tuple[str, str]]:
    return [
        ("CONVERSATION", fmt_conversation(i["conversation"])),
        ("TOOL_CALLS", fmt_tool_calls(_get(i, "tool_calls"))),
        ("AGENT_TURN", truncate(as_text(i["agent_text"]), TEXT_CAP)),
    ]


def register_builtin_kinds() -> None:
    register_kind(
        JudgeKindSpec(
            "decision",
            ("turn", "conversation", "agent_text", "tool_calls"),
            _decision_blocks,
            optional=("available_tools",),
            post_validate=_decision_post,
            family="multiclass",
        ),
        override=True,
    )
    register_kind(
        JudgeKindSpec(
            "clarification_relevance",
            ("turn", "conversation", "agent_text"),
            _clarify_blocks,
            optional=("missing_hints", "tool_schemas"),
        ),
        override=True,
    )
    register_kind(
        JudgeKindSpec(
            "semantic_arg",
            ("reference", "actual", "tool", "path"),
            _semantic_blocks,
            optional=("rubric", "user_input", "tool_description", "param_schema", "conversation"),
            orders=2,
            precheck=_semantic_precheck,
        ),
        override=True,
    )
    register_kind(
        JudgeKindSpec(
            "intent",
            ("labels", "conversation", "turn"),
            _intent_blocks,
            labels_for=_intent_labels,
            family="multiclass",
            builtin_calibratable=False,
        ),
        override=True,
    )
    register_kind(
        JudgeKindSpec(
            "refusal_alternative",
            ("turn", "conversation", "agent_text"),
            _refusal_blocks,
            optional=("tool_calls",),
        ),
        override=True,
    )


register_builtin_kinds()
