"""Builders for synthetic runs and expectations used by the assertions tests."""

from typing import Any

from disseqt_agent_testing.models import (
    AgentRun,
    RunResult,
    Step,
    ToolInfo,
    Usage,
)


def step(
    seq: int,
    name: str,
    args: dict[str, Any] | None = None,
    *,
    effect: str = "write",
    turn: int = 0,
    result: Any = None,
    kind: str = "tool",
    error: str | None = None,
    parallel_group: int | None = None,
    usage: Usage | None = None,
    duration_ms: float | None = None,
    ts: float = 0.0,
) -> Step:
    return Step(
        seq,
        kind,
        name,
        call_id=f"call-{seq}",
        args=args or {},
        result=result,
        effect=None if kind == "model" else effect,  # type: ignore[arg-type]
        turn=turn,
        error=error,
        parallel_group=parallel_group,
        usage=usage,
        duration_ms=duration_ms,
        ts=ts,
    )


def run_of(
    *steps: Step,
    text: str | None = "done",
    outputs: tuple[Any, ...] | None = None,
    tools: tuple[str, ...] = (),
    captured: bool = False,
    input: str = "refund order A12",
    follow_ups: tuple[str, ...] = (),
    offered: tuple[ToolInfo, ...] = (),
) -> AgentRun:
    return AgentRun(
        steps=tuple(steps),
        input=input,
        follow_ups=follow_ups,
        final_answer=text,
        final_text=text,
        turn_outputs=outputs if outputs is not None else (text,),
        available_tools=(*(ToolInfo(n, "tool", "write") for n in tools), *offered),
        model_steps_captured=captured,
    )


def result_of(run: AgentRun, status: str = "finished", **kw: Any) -> RunResult:
    from disseqt_agent_testing.models import ErrorInfo

    if status in ("crashed", "setup_failed"):
        kw.setdefault("error", ErrorInfo("RuntimeError", "builtins", "RuntimeError: boom"))
    return RunResult(status, run, **kw)
