"""Agents and doubles for the runner tests (plan 05 section 15)."""

import threading
import time
from typing import Any

from disseqt_agent_testing.assertions import JudgeTask, JudgeVerdict
from disseqt_agent_testing.models import (
    AgentTestCase,
    Decision,
    Expect,
    FakeResponse,
    FakeSpec,
    Ref,
    ToolExpect,
)
from disseqt_agent_testing.runner import current_repeat

ORDER = {"order_id": "A12", "total": 59.99, "eligible": True, "amount": 59.99, "status": "ok"}


def refund_fakes() -> tuple[FakeSpec, ...]:
    """The three refund fakes; every tool answers with the same order record."""
    return tuple(
        FakeSpec(
            name,
            effect="read" if name != "issue_refund" else "write",
            default=FakeResponse(returns=ORDER),
        )
        for name in ("get_order", "check_policy", "issue_refund")
    )


def refund_expect() -> Expect:
    return Expect(
        decision=Decision.CALL,
        tools=(
            ToolExpect("get_order", id="lookup"),
            ToolExpect(
                "issue_refund",
                {"order_id": Ref("call:lookup.result.order_id")},
                id="refund",
                after=("lookup",),
            ),
        ),
        order="partial",
    )


def refund_case(case_id: str = "refund_case", **kw: Any) -> AgentTestCase:
    kw.setdefault("fakes", refund_fakes())
    kw.setdefault("expect", refund_expect())
    return AgentTestCase(case_id, "refund order A12", **kw)


def make_good_agent(tools: Any) -> Any:
    def agent(text: str) -> str:
        order = tools["get_order"](order_id="A12")
        tools["issue_refund"](order_id=order["order_id"], amount=order["total"])
        return "Refunded."

    return agent


def make_patterned_agent(tools: Any, pattern: str = "PPFPF") -> Any:
    """Repeat ``i`` passes if ``pattern[i] == 'P'``; on ``F`` it refunds the wrong order."""

    def agent(text: str) -> str:
        ctx = current_repeat()
        assert ctx is not None, "current_repeat() must be set inside a repeat"
        order = tools["get_order"](order_id="A12")
        target = order["order_id"] if pattern[ctx.index % len(pattern)] == "P" else "Z999"
        tools["issue_refund"](order_id=target, amount=order["total"])
        return "Refunded."

    return agent


def patterned(pattern: str) -> Any:
    return lambda tools: make_patterned_agent(tools, pattern)


def make_slow_agent(tools: Any, delay: float = 0.1) -> Any:
    def agent(text: str) -> str:
        time.sleep(delay)
        return make_good_agent(tools)(text)  # type: ignore[no-any-return]

    return agent


def make_seed_echo_agent(tools: Any, seed: int = -1) -> Any:
    def agent(text: str) -> str:
        order = tools["get_order"](order_id="A12")
        tools["issue_refund"](order_id=order["order_id"], amount=order["total"])
        return f"seed={seed}"

    return agent


class FakeJudge:
    """Scripted ``Judge`` that records the peak number of simultaneous calls."""

    def __init__(self, label: str = "pass", delay: float = 0.0, status: str = "ok"):
        self.label, self.delay, self.status = label, delay, status
        self.calls = 0
        self._active = 0
        self.peak = 0
        self.stamps: list[float] = []
        self._lock = threading.Lock()

    def judge(self, task: JudgeTask) -> JudgeVerdict:
        with self._lock:
            self.calls += 1
            self._active += 1
            self.peak = max(self.peak, self._active)
            self.stamps.append(time.monotonic())
        try:
            time.sleep(self.delay)
            return JudgeVerdict(self.status, self.label, reason="scripted", error_code=None if self.status == "ok" else "scripted_error")  # type: ignore[arg-type]
        finally:
            with self._lock:
                self._active -= 1

    async def ajudge(self, task: JudgeTask) -> JudgeVerdict:
        return self.judge(task)


def make_wrong_agent(tools: Any) -> Any:
    """Refunds the wrong order, always (for failing-case demos)."""

    def agent(text: str) -> str:
        order = tools["get_order"](order_id="A12")
        tools["issue_refund"](order_id="Z999", amount=order["total"])
        return "Refunded."

    return agent
