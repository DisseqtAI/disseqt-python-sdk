"""Shared helpers for the harness tests."""

import functools
from typing import Any

from disseqt_agent_testing.harness import Orchestrator, build_test_orchestrator
from disseqt_agent_testing.models import RunResult
from tests.agent_testing._support.refund_agent import FAKES, make_refund_agent


def refund(**options: Any) -> Orchestrator:
    return build_test_orchestrator(make_refund_agent, FAKES, **options)


def wrong_order(**options: Any) -> Orchestrator:
    return build_test_orchestrator(
        functools.partial(make_refund_agent, bug="wrong_order"), FAKES, **options
    )


def run_body(fakes: Any, body: Any, text: str = "go", **options: Any) -> RunResult:
    """Run ``body(tools)`` as the agent against ``fakes``."""

    def factory(tools: Any) -> Any:
        return lambda t: body(tools)

    return build_test_orchestrator(factory, fakes, **options).run(text)


def normalized(result: RunResult) -> Any:
    """A run with the volatile parts (time, ids) removed, for equality checks."""
    return (
        result.status,
        result.run.final_text,
        [
            (s.seq, s.kind, s.name, s.call_id, dict(s.args), s.result, s.error)
            for s in result.run.steps
        ],
    )
