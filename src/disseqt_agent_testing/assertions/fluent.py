"""The fluent form of ``assert_agent``: sugar over ``Expect``. Each call builds a small ``Expect``
fragment and evaluates it immediately, so the first failure raises with a focused message."""

from typing import Any

from ..models import (
    AgentRun,
    Expect,
    FakeSpec,
    ForbiddenCall,
    RunResult,
    ToolExpect,
)
from .api import AgentAssertionError, evaluate, raise_for_report
from .config import EvalConfig
from .judge import Judge
from .report import DecisionReport


class AgentAssertions:
    """``assert_agent(run).called("x").before("y")``. Repeated calls: ``before`` compares the
    FIRST matching call of the earlier tool with the FIRST of the later one."""

    def __init__(
        self,
        result: RunResult | AgentRun,
        *,
        case_id: str | None = None,
        judge: Judge | None = None,
        config: EvalConfig | None = None,
        fakes: tuple[FakeSpec, ...] = (),
        options: dict[str, Any] | None = None,
    ):
        self._result = result
        self._kw: dict[str, Any] = {
            "case_id": case_id,
            "judge": judge,
            "config": config,
            "fakes": fakes,
            **(options or {}),
        }
        self.reports: list[DecisionReport] = []
        self._n = 0
        self._last: tuple[str, str, dict[str, Any]] | None = None  # (id, name, args)

    # ------------------------------------------------------------------ plumbing
    @property
    def _run(self) -> AgentRun:
        return self._result.run if isinstance(self._result, RunResult) else self._result

    def _check(self, expect: Expect) -> DecisionReport:
        __tracebackhide__ = True
        report = evaluate(self._result, expect, **self._kw)
        self.reports.append(report)
        raise_for_report(report)
        return report

    def _ran(self, name: str) -> list[int]:
        return [s.seq for s in self._run.tool_steps() if s.name == name]

    # ------------------------------------------------------------------ assertions
    def called(self, name: str, **args: Any) -> "AgentAssertions":
        __tracebackhide__ = True
        self._n += 1
        eid = f"called_{self._n}"
        self._last = (eid, name, args)
        self._check(
            Expect(tools=(ToolExpect(name, args, id=eid, count=(1, None)),), extra_calls="allow")
        )
        return self

    def once(self) -> "AgentAssertions":
        __tracebackhide__ = True
        eid, name, args = self._require_last("once")
        self._check(Expect(tools=(ToolExpect(name, args, id=eid, count=1),), extra_calls="allow"))
        return self

    def before(self, other: str, **args: Any) -> "AgentAssertions":
        __tracebackhide__ = True
        eid, name, first_args = self._require_last("before")
        order = [s.name for s in self._run.tool_steps()]
        for tool in (name, other):
            if not self._ran(tool):
                raise AgentAssertionError(
                    f"tool_order: expected {name} before {other}; {tool} never ran "
                    f"(observed {' -> '.join(order) or 'no calls'})",
                    self.reports[-1] if self.reports else _empty_report(self._kw["case_id"]),
                )
        self._n += 1
        oid = f"called_{self._n}"
        expect = Expect(
            tools=(
                ToolExpect(name, first_args, id=eid, count=(1, None)),
                ToolExpect(other, args, id=oid, count=(1, None), after=(eid,)),
            ),
            order="partial",
            extra_calls="allow",
        )
        report = evaluate(self._result, expect, **self._kw)
        self.reports.append(report)
        order_check = report.check("order")
        if order_check is not None and order_check.passed is False:
            raise AgentAssertionError(
                f"tool_order: expected {name} before {other}; observed {' -> '.join(order)}\n"
                + report.render(),
                report,
            )
        raise_for_report(report)
        self._last = (oid, other, args)
        return self

    def never_called(self, name: str, **args: Any) -> "AgentAssertions":
        __tracebackhide__ = True
        self._check(Expect(forbidden=(ForbiddenCall(name, args or None),)))
        return self

    def stayed_within(self, **budgets: Any) -> "AgentAssertions":
        __tracebackhide__ = True
        self._check(Expect(**budgets))
        return self

    def _require_last(self, method: str) -> tuple[str, str, dict[str, Any]]:
        if self._last is None:
            raise ValueError(f".{method}() needs a preceding .called(...)")
        return self._last


def _empty_report(case_id: str | None) -> DecisionReport:
    return DecisionReport(case_id or "case", "failed")
