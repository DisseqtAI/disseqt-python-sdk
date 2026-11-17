"""Phase-4b kinds: answer_quality, groundedness, plan_quality, root_cause.

The first three decide their label in code from structured model output (per-criterion booleans,
per-claim verdicts, a list of flaws); the model never self-reports the final verdict.
``root_cause`` is advisory and never gates.
"""

from collections.abc import Mapping
from typing import Any

from ..render import TEXT_CAP, as_text, fmt_conversation, fmt_list, fmt_tool_calls, truncate
from .base import JudgeKindSpec, Precheck, register_kind

ROOT_CAUSES = (
    "wrong_tool", "wrong_args", "missing_clarification", "hallucinated_answer",
    "prompt_ambiguity", "tool_failure", "loop", "other",
)  # fmt: skip
FLAW_CODES = ("missing_step", "wrong_order", "redundant_step", "wrong_tool", "unsafe_step", "other")
CLAIM_VERDICTS = ("supported", "unsupported", "contradicted", "not_checkable")


def _criteria(i: Mapping[str, Any]) -> list[str]:
    return [str(c) for c in i["criteria"]]


# ----------------------------------------------------------------------------- answer_quality
def _answer_check(i: Mapping[str, Any]) -> str | None:
    criteria = i["criteria"]
    if not isinstance(criteria, (list, tuple)) or not [c for c in criteria if str(c).strip()]:
        return "criteria must be a non-empty list: with none, nothing can be judged"
    return None


def _answer_blocks(i: Mapping[str, Any], order: int, sample: int) -> list[tuple[str, str]]:
    return [
        ("CRITERIA", "\n".join(f"{n}. {c}" for n, c in enumerate(_criteria(i), 1))),
        ("CONVERSATION", fmt_conversation(i["conversation"])),
        ("ANSWER", truncate(as_text(i["agent_text"]), TEXT_CAP)),
    ]


def _answer_post(parsed: Mapping[str, Any], i: Mapping[str, Any]) -> tuple[str, str] | None:
    ids = sorted(int(c["id"]) for c in parsed.get("criteria", ()))
    want = list(range(1, len(_criteria(i)) + 1))
    if ids != want:
        return "invalid_verdict", f"criteria ids {ids} do not match the {len(want)} given criteria"
    return None


def _answer_derive(parsed: Mapping[str, Any], i: Mapping[str, Any]) -> str:
    if parsed.get("enough_data") is False:
        return "unsure"
    return "pass" if all(c["met"] for c in parsed["criteria"]) else "fail"


def _answer_summary(parsed: Mapping[str, Any]) -> Any:
    return {"criteria": [{"id": c["id"], "met": c["met"]} for c in parsed["criteria"]]}


# ----------------------------------------------------------------------------- groundedness
def _ground_blocks(i: Mapping[str, Any], order: int, sample: int) -> list[tuple[str, str]]:
    return [
        ("SOURCES", fmt_tool_calls(i["tool_calls"])),
        ("CONTEXT", as_text(i.get("context") or "(none)")),
        ("ANSWER", truncate(as_text(i["agent_text"]), TEXT_CAP)),
    ]


def _ground_derive(parsed: Mapping[str, Any], i: Mapping[str, Any]) -> str:
    if parsed.get("enough_data") is False:
        return "unsure"
    claims = parsed["claims"]
    if not claims:  # no checkable claim from a non-blank answer is the judge failing, not a pass
        return "unsure"
    bad = [c for c in claims if c["verdict"] in ("unsupported", "contradicted")]
    return "ungrounded" if bad else "grounded"


def _ground_precheck(i: Mapping[str, Any]) -> Precheck | None:
    """A blank answer asserts nothing: the one case ``not_applicable`` is justified in code."""
    if not as_text(i["agent_text"]).strip():
        return Precheck("not_applicable", "the answer is empty: no claim to ground", "empty_answer")
    return None


def _ground_summary(parsed: Mapping[str, Any]) -> Any:
    claims = parsed["claims"]
    counts = {v: sum(1 for c in claims if c["verdict"] == v) for v in CLAIM_VERDICTS}
    return {"claims": [{"claim": c["claim"][:200], "verdict": c["verdict"]} for c in claims],
            "counts": counts}  # fmt: skip


# ----------------------------------------------------------------------------- plan_quality
def _plan_blocks(i: Mapping[str, Any], order: int, sample: int) -> list[tuple[str, str]]:
    return [
        ("GOAL", truncate(as_text(i["goal"]), TEXT_CAP)),
        ("TOOLS", fmt_list(i.get("available_tools"))),
        ("PLAN", fmt_list(i["plan"])),
        ("TOOL_CALLS", fmt_tool_calls(i.get("tool_calls"))),
    ]


def _plan_derive(parsed: Mapping[str, Any], i: Mapping[str, Any]) -> str:
    # An empty ``flaws`` list is the judge's call ("sound"); no code rule second-guesses it.
    if parsed.get("enough_data") is False:
        return "unsure"
    return "flawed" if parsed["flaws"] else "sound"


def _plan_summary(parsed: Mapping[str, Any]) -> Any:
    return {"flaws": [{"code": f["code"], "step": f["step"]} for f in parsed["flaws"]]}


# ----------------------------------------------------------------------------- root_cause
def _root_blocks(i: Mapping[str, Any], order: int, sample: int) -> list[tuple[str, str]]:
    return [
        ("FAILED_CHECKS", fmt_list(i["failed_checks"])),
        ("CONVERSATION", fmt_conversation(i["conversation"])),
        ("TOOL_CALLS", fmt_tool_calls(i.get("tool_calls"))),
        ("AGENT_TURN", truncate(as_text(i.get("agent_text", "")), TEXT_CAP)),
    ]


def register_quality_kinds() -> None:
    register_kind(JudgeKindSpec(
        "answer_quality", ("conversation", "agent_text", "criteria"), _answer_blocks,
        check_inputs=_answer_check, post_validate=_answer_post, derive=_answer_derive, summarize=_answer_summary, positive="pass",
    ), override=True)  # fmt: skip
    register_kind(JudgeKindSpec(
        "groundedness", ("agent_text", "tool_calls"), _ground_blocks, optional=("context",),
        precheck=_ground_precheck, derive=_ground_derive, summarize=_ground_summary,
        positive="grounded",
    ), override=True)  # fmt: skip
    register_kind(JudgeKindSpec(
        "plan_quality", ("goal", "plan"), _plan_blocks, optional=("available_tools", "tool_calls"),
        derive=_plan_derive, summarize=_plan_summary, positive="sound",
    ), override=True)  # fmt: skip
    register_kind(JudgeKindSpec(
        "root_cause", ("failed_checks", "conversation"), _root_blocks,
        optional=("tool_calls", "agent_text"), never_gates=True, family="multiclass",
        builtin_calibratable=False,
    ), override=True)  # fmt: skip


register_quality_kinds()
