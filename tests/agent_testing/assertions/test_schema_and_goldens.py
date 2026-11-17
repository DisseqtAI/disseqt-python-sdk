"""3c: the JSON Schema golden, a golden report, conformance and performance guards."""

import dataclasses
import json
import statistics
import time
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.assertions import DecisionReport, evaluate
from disseqt_agent_testing.assertions.report import report_json_schema
from disseqt_agent_testing.harness import build_test_orchestrator
from disseqt_agent_testing.models import Decision, Expect, ToolExpect
from tests.agent_testing._support.refund_agent import FAKES, make_refund_agent

from .._support.schema_check import validate
from .helpers import run_of, step
from .test_integration import REFUND, wrong_order

GOLDEN = Path(__file__).parent / "golden"


def test_the_json_schema_matches_its_golden_file() -> None:
    stored = json.loads((GOLDEN / "report_schema_v1.json").read_text())
    assert (
        report_json_schema() == stored
    ), "regenerate report_schema_v1.json only for an additive change"


def test_the_schema_checker_rejects_bad_documents() -> None:
    schema = report_json_schema()
    assert validate({}, schema, schema)  # missing everything
    rep = evaluate(run_of(step(1, "a")), Expect(tools=(ToolExpect("a"),))).to_dict()
    rep["outcome"] = "fine"
    rep["checks"][0]["value"] = 7
    problems = validate(rep, schema, schema)
    assert any("outcome" in p for p in problems) and any("maximum" in p for p in problems)


def reports() -> list[DecisionReport]:
    ok = build_test_orchestrator(make_refund_agent, FAKES).run("refund order A12")
    crashed = build_test_orchestrator(lambda tools: (lambda t: 1 / 0), FAKES).run("go")
    out = [
        evaluate(ok, REFUND),
        evaluate(wrong_order(), REFUND),
        evaluate(
            run_of(text="Which one?"), Expect(decision=Decision.CALL, tools=(ToolExpect("a"),))
        ),
        evaluate(run_of(step(1, "a")), Expect(max_llm_calls=1)),
        evaluate(crashed, Expect(tools=(ToolExpect("a"),))),
    ]
    return out


@pytest.mark.parametrize("index", range(5))
def test_every_kind_of_report_conforms_to_the_schema(index: int) -> None:
    schema = report_json_schema()
    data = json.loads(json.dumps(reports()[index].to_dict()))
    assert validate(data, schema, schema) == []
    assert DecisionReport.from_dict(data) == reports()[index].__class__.from_dict(data)


def deterministic_report() -> DecisionReport:
    ticks = iter(range(0, 10**12, 1_000_000))
    orch = build_test_orchestrator(
        __import__("functools").partial(make_refund_agent, bug="wrong_order"),
        FAKES,
        clock=lambda: next(ticks),
        id_factory=lambda: "golden-trace-id",
    )
    result = orch.run("refund order A12")
    result = dataclasses.replace(result, duration_ms=0.0, setup_ms=0.0)
    return evaluate(result, REFUND, case_id="refund_wrong_order")


def test_a_golden_report_is_stable_and_loads_on_later_versions() -> None:
    path = GOLDEN / "report_v1.json"
    stored = json.loads(path.read_text())
    assert deterministic_report().to_dict() == stored, "re-generate only for a deliberate change"
    loaded = DecisionReport.from_dict(stored)
    assert loaded.to_dict() == stored  # old content preserved, in order
    assert loaded.outcome == "failed" and loaded.check("tool_args").reason_code == "ungrounded_value"  # type: ignore[union-attr]


def median_ms(fn: Any, n: int = 25) -> float:
    fn()
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def test_i09_nfr04_performance_stays_within_guard_rails() -> None:
    """Measured on a laptop: 50 calls x 20 expectations ~2.3-2.6 ms, a typical 10-call run
    ~0.56 ms, render ~0.3 ms (targets: 5 ms, 0.5 ms, 2 ms). The guards are 4x looser so a busy CI
    runner does not flake; a real regression (an accidental O(n^3)) still trips them."""
    steps = [step(i + 1, f"t{i % 10}", {"a": i, "b": "x"}) for i in range(50)]
    big = run_of(*steps)
    tools = tuple(ToolExpect(f"t{i % 10}", {"a": i}, id=f"e{i}") for i in range(20))
    heavy = Expect(
        decision=Decision.CALL, tools=tools, order="in_order", extra_calls="allow", max_steps=100
    )
    assert median_ms(lambda: evaluate(big, heavy)) < 20.0
    small = run_of(*[step(i + 1, f"t{i % 3}", {"a": i}) for i in range(10)])
    typical = Expect(
        decision=Decision.CALL,
        tools=tuple(ToolExpect(f"t{i % 3}", {"a": i}) for i in range(4)),
        extra_calls="allow",
    )
    assert median_ms(lambda: evaluate(small, typical)) < 4.0
    rep = evaluate(big, heavy)
    assert median_ms(lambda: rep.render(max_lines=0)) < 8.0


def test_evaluation_cost_grows_roughly_with_the_product_not_worse() -> None:
    def cost(n_calls: int, n_expect: int) -> float:
        steps = [step(i + 1, f"t{i % 10}", {"a": i}) for i in range(n_calls)]
        tools = tuple(ToolExpect(f"t{i % 10}", {"a": i}, id=f"e{i}") for i in range(n_expect))
        expect = Expect(tools=tools, extra_calls="allow")
        run = run_of(*steps)
        return median_ms(lambda: evaluate(run, expect), n=9)

    small, large = cost(10, 5), cost(40, 20)  # 8x the pairs
    assert large < small * 8 * 6  # generous: guards against super-linear blow-ups
