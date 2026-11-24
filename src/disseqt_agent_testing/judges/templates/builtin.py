"""The v1.0 templates. Each is the common preamble plus a task text; no few-shots in v1.0."""

from . import PREAMBLE, JudgeTemplate, register_template

REASONING = {"type": "string"}


def _schema(label_enum: list[str], extra: dict[str, object] | None = None) -> dict[str, object]:
    props: dict[str, object] = {
        "reasoning": REASONING,
        "label": {"type": "string", "enum": label_enum},
    }
    props.update(extra or {})
    return {
        "type": "object", "properties": props, "required": list(props),
        "additionalProperties": False,
    }  # fmt: skip


DECISION_TASK = (
    "TASK: Classify the DECISION the agent made in <AGENT_TURN>. Do NOT judge whether the decision "
    "was correct, safe or accurate. Labels: call (the agent made tool calls this turn, listed in "
    "<TOOL_CALLS>, and its text only reports or summarizes that work); ask (asks the user for "
    "information or confirmation it needs to proceed, and does not decline); refuse (declines to do "
    "what was requested, with or without an alternative); answer (provides the requested content "
    "directly, without asking for needed input and without declining); unsure (empty, garbled, or "
    "mixes decisions with no dominant one). Tie-breaks: declines AND asks: choose ask if answering "
    "the question would let the agent complete the original request, otherwise refuse; a closing "
    "courtesy question never makes a turn ask; choose call only if <TOOL_CALLS> is not empty."
)
CLARIFY_TASK = (
    "TASK: The agent asked the user a question in <AGENT_TURN>. Decide whether it targets "
    "information genuinely missing or ambiguous AND needed. pass: asks for at least one missing or "
    "ambiguous needed item (for example a required parameter in <TOOL_SCHEMAS> that cannot be "
    "inferred) and is not only asking for what the user already gave. fail: asks only for provided, "
    "unneeded or unrelated information. not_applicable: no question to the user. unsure: "
    "insufficient data. <MISSING_HINTS> lists items the author expects to be missing; a needed "
    "missing item not listed still passes. Tie-breaks: needed missing item plus a re-asked provided "
    "item: pass, noting redundancy; borderline with sufficient data: fail."
)
SEMANTIC_TASK = (
    "TASK: Decide whether two values for the same tool argument are semantically equivalent FOR THIS "
    "TOOL CALL: would a correct implementation of the tool behave the same way with <VALUE_A> as "
    "with <VALUE_B>? equivalent: only formatting, spelling, case, abbreviation, synonyms, word order "
    "or unit-preserving notation differ. not_equivalent: a different entity, quantity, unit, "
    "date/time, range, scope or negation. unsure: depends on missing context. <ARG_RUBRIC> (the "
    "author's definition) overrides the guidance. Resolve relative expressions only against dates "
    "stated in <CONVERSATION>; otherwise a relative expression is equivalent only to the same "
    "relative expression. VALUE_A and VALUE_B are interchangeable."
)
INTENT_TASK = (
    "TASK: Classify the user's PRIMARY intent at turn <TURN> of <CONVERSATION> into exactly one "
    "label from <LABELS>; judge only what the user wants, ignore what the agent did. other: none "
    "fits. unsure: too ambiguous between labels. Tie-break: several goals: the one asked first or "
    "explicitly prioritised; none prioritised: unsure. Label order carries no meaning."
)
REFUSAL_TASK = (
    "TASK: The agent declined or could not fully complete the request. Did it offer at least one "
    "concrete, relevant alternative? pass: a specific next step (different action or tool, narrower "
    "request, specific resource or channel, hand-off). fail: no alternative or only generic "
    'statements ("Sorry, I can\'t help", "Contact support" without how). not_applicable: the agent '
    "did not decline. unsure. Tie-break: vague or borderline counts as fail. List alternatives in "
    "<alternatives>."
)


def _t(
    kind: str,
    task: str,
    blocks: tuple[str, ...],
    labels: list[str],
    values: dict[str, float | None],
    extra: dict[str, object] | None = None,
) -> JudgeTemplate:
    return JudgeTemplate(
        kind, "1.0", f"{PREAMBLE}\n{task}", blocks, tuple(labels), _schema(labels, extra),
        value_map=values, builtin=True,
    )  # fmt: skip


_STR_LIST = {"type": "array", "items": {"type": "string"}}
TEMPLATES = (
    _t(
        "decision",
        DECISION_TASK,
        ("AVAILABLE_TOOLS", "CONVERSATION", "TOOL_CALLS", "AGENT_TURN"),
        ["call", "ask", "refuse", "answer", "unsure"],
        {},
    ),
    _t(
        "clarification_relevance",
        CLARIFY_TASK,
        ("TOOL_SCHEMAS", "MISSING_HINTS", "CONVERSATION", "AGENT_TURN"),
        ["pass", "fail", "not_applicable", "unsure"],
        {},
        {"items_requested": _STR_LIST, "asks_for_known_info": {"type": "boolean"}},
    ),
    _t(
        "semantic_arg",
        SEMANTIC_TASK,
        ("TOOL", "ARG_RUBRIC", "CONVERSATION", "VALUE_A", "VALUE_B"),
        ["equivalent", "not_equivalent", "unsure"],
        {"equivalent": 1.0, "not_equivalent": 0.0},
    ),
    _t("intent", INTENT_TASK, ("LABELS", "CONVERSATION", "TURN"), ["other", "unsure"], {}),
    _t(
        "refusal_alternative",
        REFUSAL_TASK,
        ("CONVERSATION", "TOOL_CALLS", "AGENT_TURN"),
        ["pass", "fail", "not_applicable", "unsure"],
        {},
        {"alternatives": _STR_LIST},
    ),
)
for _template in TEMPLATES:
    register_template(_template, override=True)
