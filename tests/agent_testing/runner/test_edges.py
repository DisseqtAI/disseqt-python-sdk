"""Edge behaviour of the runner: async error paths, JSON parsing, config validation, unknown codes."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.harness import HarnessConfigError
from disseqt_agent_testing.models import Decision, Expect, ToolExpect
from disseqt_agent_testing.runner import (
    CODES,
    CaseSpec,
    CaseStarted,
    RunBlock,
    SuiteConfig,
    SuiteConfigError,
    arun_case,
    arun_suite,
    format_code,
    load_cases,
    run_suite,
)
from disseqt_agent_testing.runner._json import JsonProblem, parse_json
from disseqt_agent_testing.runner.loader import discover, format_errors

from .._support.runner_agents import (
    FakeJudge,
    make_good_agent,
    refund_case,
    refund_expect,
    refund_fakes,
)


def test_unknown_error_codes_render_a_generic_line_and_known_ones_their_text() -> None:
    assert (
        format_code("DSQ-X99", a=1) == "DSQ-X99: {'a': 1}" and format_code("DSQ-X99") == "DSQ-X99"
    )
    assert format_code("DSQ-S04") == "fail-fast: an earlier case failed"
    assert format_code("DSQ-S03").startswith("DSQ-S03")  # a missing field degrades, it never raises
    assert "DSQ-C01" in CODES and all(c.startswith("DSQ-") for c in CODES)


def test_every_documented_message_has_a_stable_code() -> None:
    expected = {"DSQ-C01", "DSQ-C02", "DSQ-C03", "DSQ-C04", "DSQ-C05", "DSQ-C06", "DSQ-C07", "DSQ-C08", "DSQ-C09",
                "DSQ-C10", "DSQ-C11", "DSQ-E01", "DSQ-E02", "DSQ-E03", "DSQ-E04", "DSQ-E05", "DSQ-E06", "DSQ-E07",
                "DSQ-F01", "DSQ-F02", "DSQ-I01", "DSQ-S01", "DSQ-S02", "DSQ-S03", "DSQ-S04", "DSQ-S05", "DSQ-U01",
                "DSQ-U02", "DSQ-U03"} | {f"DSQ-W{n:02d}" for n in range(1, 12)}  # fmt: skip
    assert expected <= set(CODES)


def test_json_parser_handles_every_value_kind_and_reports_lines() -> None:
    value, lines = parse_json(
        '{\n "a": [1, 2.5e1, -3, true, false, null, "x\\ny"],\n "b": {}, "c": []\n}'
    )
    assert value == {"a": [1, 25.0, -3, True, False, None, "x\ny"], "b": {}, "c": []}
    assert lines[("a",)] == 2 and lines[("b",)] == 3 and lines[("a", 6)] == 2
    for bad, text in [
        ('{"a": 1,}', "property name"),
        ('{"a": tru}', "Expecting value"),
        ('"unterminated', "Unterminated"),
        ("[1,", "Expecting value"),
        ("-", "Invalid number"),
        ("{1: 2}", "property name"),
    ]:
        with pytest.raises(JsonProblem) as err:
            parse_json(bad)
        assert text.lower() in err.value.problem.lower(), (bad, err.value.problem)


def test_config_validation_covers_every_field() -> None:
    for bad in ({"run_timeout_s": -1}, {"case_timeout_s": 0}, {"suite_timeout_s": True}, {"judge_concurrency": -1},
                {"repeat_mode": "x"}, {"eval_config": 5}, {"default_repeats": 0}, {"repeats": True}):  # fmt: skip
        with pytest.raises(SuiteConfigError):
            SuiteConfig(**bad)


def test_the_pool_and_engine_survive_a_unit_that_raises_an_ordinary_exception() -> None:
    def factory(tools: Any) -> Any:
        def agent(text: str) -> str:
            raise RuntimeError("agent bug")

        return agent

    c = run_suite([refund_case(repeats=2)], agent_factory=factory).cases[0]
    assert c.status == "failed" and {r.run_status for r in c.repeats} == {"crashed"}


def test_discover_ignores_non_matching_files_and_accepts_a_single_file(tmp_path: Path) -> None:
    (tmp_path / "a.agent.json").write_text("{}")
    (tmp_path / "b.txt").write_text("")
    assert [p.name for p in discover(tmp_path)] == ["a.agent.json"]
    assert discover(tmp_path / "a.agent.json") == [tmp_path / "a.agent.json"]


def test_loader_error_shapes(tmp_path: Path) -> None:
    def errs(text: str, name: str = "x.agent.json") -> list[str]:
        (tmp_path / name).write_text(text)
        return [e.message for e in load_cases(tmp_path / name).errors]

    assert "must be a mapping" in errs("[]")[0]
    assert "expected a list of cases" in errs('{"version": 1, "cases": 5}')[0]
    assert "expected a mapping" in errs('{"version": 1, "defaults": 3, "cases": []}')[0]
    assert "expected a mapping" in errs('{"version": 1, "cases": [5]}')[0]
    assert (
        "rows must be a list of mappings"
        in errs('{"version": 1, "cases": [{"id": "t_${a}", "input": "x", "rows": [1]}]}')[0]
    )
    assert (
        "unsupported format"
        in errs('{"version": 1, "cases": [{"id": "t_${a}", "input": "x", "dataset": "rows.xyz"}]}')[
            0
        ]
    )
    assert (
        "pass_k_threshold"
        in errs('{"version": 1, "cases": [{"id": "a", "input": "x", "pass_k_threshold": "high"}]}')[
            0
        ]
    )
    (tmp_path / "empty.agent.json").write_text('{"version": 1, "cases": []}')
    assert any(e.code == "DSQ-W10" for e in load_cases(tmp_path / "empty.agent.json").errors)
    assert format_errors([]) == []
    spec = load_cases(tmp_path / "missing_dir").errors
    assert spec and "no such file" in spec[0].message


# --------------------------------------------------------------------- async twins
def test_async_engine_error_classification_and_events() -> None:
    kinds: list[str] = []
    r = asyncio.run(
        arun_suite(
            [refund_case("e1"), CaseSpec(refund_case("e2"), run=RunBlock(skip="later"))],
            agent_factory=lambda tools, db: None,
            on_event=lambda e: kinds.append(type(e).__name__),
        )
    )
    by = {c.case_id: c for c in r.cases}
    assert by["e1"].reason_code == "harness_config" and by["e2"].status == "skipped"
    assert "CaseFinished" in kinds and kinds[-1] == "SuiteFinished"


def test_async_expect_config_error_and_unexpected_exception() -> None:
    bad = refund_case(
        "x", expect=Expect(tools=refund_expect().tools, order="partial", min_scores={"nope": 1.0})
    )
    assert asyncio.run(arun_case(bad, agent_factory=make_good_agent)).reason_code == "expect_config"


def test_async_only_repeat_early_stop_and_fail_fast() -> None:
    from .._support.runner_agents import make_wrong_agent

    one = asyncio.run(
        arun_case(
            refund_case("o", repeats=3),
            agent_factory=make_good_agent,
            config=SuiteConfig(only_repeat=1),
        )
    )
    assert [r.status for r in one.repeats] == ["not_run", "passed", "not_run"]
    stopped = asyncio.run(
        arun_case(
            refund_case("s", repeats=6),
            agent_factory=make_wrong_agent,
            config=SuiteConfig(early_stop=True, max_workers=1),
        )
    )
    assert stopped.status == "failed" and stopped.stopped_early
    ff = asyncio.run(
        arun_suite(
            [refund_case("a_bad"), refund_case("b_ok"), refund_case("c_ok")],
            agent_factory=make_wrong_agent,
            config=SuiteConfig(fail_fast=True, max_workers=1),
        )
    )
    assert any(c.reason_code == "fail_fast" for c in ff.cases)


def test_async_cancellation_inside_a_unit_is_reraised() -> None:
    async def slow(tools: Any) -> Any:
        async def agent(text: str) -> str:
            await asyncio.sleep(10)
            return "x"

        return agent

    async def scenario() -> None:
        task = asyncio.ensure_future(
            arun_suite(
                [refund_case("c", repeats=2)],
                agent_factory=slow,
                config=SuiteConfig(run_timeout_s=30),
            )
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_async_judge_is_used_with_the_limiter() -> None:
    from disseqt_agent_testing.assertions import EvalConfig
    from disseqt_agent_testing.models import ClarifyExpect

    case = refund_case(
        "j",
        expect=Expect(clarify=ClarifyExpect(judge_relevance=True)),
        use_judge=("clarification_relevance",),
        repeats=3,
    )
    judge = FakeJudge("pass")
    cfg = SuiteConfig(eval_config=EvalConfig(use_judge=True), judge_concurrency=2)
    r = asyncio.run(
        arun_case(
            case,
            agent_factory=lambda tools: (lambda text: "Which order number is this about?"),
            judge=judge,
            config=cfg,
        )
    )
    assert judge.calls == 3 and r.status in ("passed", "incomplete")


def test_the_sync_engine_handles_a_stuck_case_with_the_grace_period() -> None:
    cfg = SuiteConfig(max_workers=1, case_timeout_s=0.15, run_timeout_s=30)

    def factory(tools: Any) -> Any:
        import time

        return lambda text: time.sleep(3)

    import time as _t

    start = _t.monotonic()
    result = run_suite([refund_case("stuck", repeats=2)], agent_factory=factory, config=cfg)
    assert _t.monotonic() - start < 6 and result.cases[0].status in (
        "skipped",
        "incomplete",
        "failed",
    )
    assert all(r.status == "not_run" or r.run_status for r in result.cases[0].repeats)


_ = (CaseStarted, HarnessConfigError, ToolExpect, Decision, refund_fakes, json)
