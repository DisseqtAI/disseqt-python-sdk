"""v1.0 templates of the phase-4b kinds. Output schemas carry structured evidence; the final
label of answer_quality / groundedness / plan_quality is derived in code, never self-reported."""

from . import PREAMBLE, JudgeTemplate, register_template
from .builtin import REASONING

_BOOL = {"type": "boolean"}


def _obj(props: dict[str, object]) -> dict[str, object]:
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}  # fmt: skip


def _items(props: dict[str, object]) -> dict[str, object]:
    return {"type": "array", "items": _obj(props)}


ANSWER_TASK = (
    "TASK: Check <ANSWER> against each numbered criterion in <CRITERIA>, using only <CONVERSATION> "
    "and <ANSWER>. For every criterion give its number as id and met=true only if the answer clearly "
    "satisfies it. Do not add criteria of your own. Set enough_data=false only if the data is too "
    "thin to check any criterion."
)
GROUND_TASK = (
    "TASK: List the factual claims <ANSWER> makes (at most 12, the most consequential first) and "
    "check each against <SOURCES> and <CONTEXT>. supported: stated or directly implied by a "
    "source. unsupported: not found in any source. contradicted: a source states otherwise. "
    "not_checkable: an opinion, greeting or a statement about the agent itself. Set "
    "enough_data=false if the sources are missing or unreadable."
)
PLAN_TASK = (
    "TASK: Review <PLAN> for achieving <GOAL> with <TOOLS>. List each real flaw with the 1-based "
    "step it concerns (0 for the plan as a whole): missing_step, wrong_order, redundant_step, "
    "wrong_tool, unsafe_step, other. Style preferences are not flaws; an empty list means the plan is "
    "sound. Set enough_data=false if the goal or plan is too vague to review."
)
ROOT_TASK = (
    "TASK: Explain the most likely root cause of the failed checks in <FAILED_CHECKS>. Labels: "
    "wrong_tool, wrong_args, missing_clarification, hallucinated_answer, prompt_ambiguity, "
    "tool_failure, loop, other, unsure. This is advisory triage only."
)

_ANSWER_SCHEMA = _obj({
    "reasoning": REASONING,
    "criteria": _items({"id": {"type": "integer"}, "met": _BOOL}),
    "enough_data": _BOOL,
})  # fmt: skip
_GROUND_SCHEMA = _obj({
    "reasoning": REASONING,
    "claims": _items({"claim": {"type": "string"},
                      "verdict": {"type": "string", "enum": ["supported", "unsupported",
                                                             "contradicted", "not_checkable"]}}),
    "enough_data": _BOOL,
})  # fmt: skip
_PLAN_SCHEMA = _obj({
    "reasoning": REASONING,
    "flaws": _items({"code": {"type": "string", "enum": ["missing_step", "wrong_order",
                                                         "redundant_step", "wrong_tool",
                                                         "unsafe_step", "other"]},
                     "step": {"type": "integer"}}),
    "enough_data": _BOOL,
})  # fmt: skip
_ROOT_LABELS = ["wrong_tool", "wrong_args", "missing_clarification", "hallucinated_answer",
                "prompt_ambiguity", "tool_failure", "loop", "other", "unsure"]  # fmt: skip
_ROOT_SCHEMA = _obj({"reasoning": REASONING, "label": {"type": "string", "enum": _ROOT_LABELS}})

TEMPLATES = (
    JudgeTemplate("answer_quality", "1.0", f"{PREAMBLE}\n{ANSWER_TASK}",
                  ("CRITERIA", "CONVERSATION", "ANSWER"), ("pass", "fail", "unsure"),
                  _ANSWER_SCHEMA, value_map={"pass": 1.0, "fail": 0.0}, builtin=True),
    JudgeTemplate("groundedness", "1.0", f"{PREAMBLE}\n{GROUND_TASK}",
                  ("SOURCES", "CONTEXT", "ANSWER"),
                  ("grounded", "ungrounded", "not_applicable", "unsure"), _GROUND_SCHEMA,
                  value_map={"grounded": 1.0, "ungrounded": 0.0}, builtin=True),
    JudgeTemplate("plan_quality", "1.0", f"{PREAMBLE}\n{PLAN_TASK}",
                  ("GOAL", "TOOLS", "PLAN", "TOOL_CALLS"), ("sound", "flawed", "unsure"),
                  _PLAN_SCHEMA, value_map={"sound": 1.0, "flawed": 0.0}, builtin=True),
    JudgeTemplate("root_cause", "1.0", f"{PREAMBLE}\n{ROOT_TASK}",
                  ("FAILED_CHECKS", "CONVERSATION", "TOOL_CALLS", "AGENT_TURN"),
                  tuple(_ROOT_LABELS), _ROOT_SCHEMA, builtin=True),
)  # fmt: skip
for _t in TEMPLATES:
    register_template(_t, override=True)
