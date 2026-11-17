"""Build the frozen judge-input shapes (plan 04 section 10.2) from a run."""

from typing import Any

from ..models import AgentRun, Message, thaw
from .classify import turn_calls


def conversation(
    run: AgentRun, upto_turn: int, *, include_agent_of_turn: bool = False
) -> list[dict[str, Any]]:
    """``[{"role", "turn", "text"}, ...]`` up to the user message of ``upto_turn`` (and, when
    asked, the agent's own reply on that turn)."""
    out: list[dict[str, Any]] = []
    if isinstance(run.input, str):
        out.append({"role": "user", "turn": 0, "text": run.input})
    else:
        for m in run.input:
            if isinstance(m, Message):
                out.append({"role": m.role, "turn": 0, "text": m.content})
    n_turns = max(len(run.turn_outputs), len(run.follow_ups) + 1)
    for t in range(n_turns):
        if t > 0 and t - 1 < len(run.follow_ups):
            if t > upto_turn:
                break
            out.append({"role": "user", "turn": t, "text": run.follow_ups[t - 1]})
        if t < upto_turn or (t == upto_turn and include_agent_of_turn):
            reply = run.turn_outputs[t] if t < len(run.turn_outputs) else None
            if reply is not None:
                out.append(
                    {
                        "role": "assistant",
                        "turn": t,
                        "text": reply if isinstance(reply, str) else str(thaw(reply)),
                    }
                )
    return out


def tool_calls(run: AgentRun, turn: int) -> list[dict[str, Any]]:
    return [
        {
            "call_id": s.call_id, "name": s.name, "args": thaw(s.args), "result": thaw(s.result),
            "error": s.error, "effect": s.effect,
        }
        for s in turn_calls(run, turn)
    ]  # fmt: skip


def tool_schemas(run: AgentRun) -> dict[str, Any]:
    return {t.name: thaw(t.schema) for t in run.available_tools if t.schema}
