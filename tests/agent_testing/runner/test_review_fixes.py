"""Review fix pass: timeouts are incomplete (never skipped), all-skipped exits 5, merge rules,
dataset confinement."""

import dataclasses
from pathlib import Path
from typing import Any

from disseqt_agent_testing.runner import CaseResult, SuiteConfig, run_suite
from disseqt_agent_testing.runner.merge import merge_repeat_results
from disseqt_agent_testing.runner.models import compute_exit_code

from .._support.runner_agents import make_good_agent, make_slow_agent, refund_case


def test_a_suite_timeout_leaves_unstarted_cases_incomplete_and_exits_2() -> None:
    cfg = SuiteConfig(max_workers=1, suite_timeout_s=0.25)
    cases = [refund_case(f"c{i}", repeats=1) for i in range(6)]
    result = run_suite(cases, agent_factory=lambda tools: make_slow_agent(tools, 0.2), config=cfg)
    cut = [c for c in result.cases if c.reason_code == "suite_timeout"]
    assert cut and all(c.status == "incomplete" for c in cut)
    assert not any(c.status == "skipped" for c in result.cases)
    assert result.exit_code == 2


def test_a_timeout_stays_exit_2_even_when_the_incomplete_policy_is_skip() -> None:
    cut = CaseResult("a", "incomplete", reason_code="suite_timeout", reason="x", n_planned=1)
    assert compute_exit_code((cut,), (), interrupted=False, incomplete_policy="skip") == 2
    plain = CaseResult("b", "incomplete", reason_code="undecided", reason="x", n_planned=1)
    assert compute_exit_code((plain,), (), interrupted=False, incomplete_policy="skip") == 0


def test_a_case_cancelled_before_it_ran_is_incomplete() -> None:
    cfg = SuiteConfig(max_workers=1, case_timeout_s=0.01, run_timeout_s=5)
    result = run_suite(
        [refund_case("slow", repeats=3)],
        agent_factory=lambda tools: make_slow_agent(tools, 0.2),
        config=cfg,
    )
    assert result.cases[0].status in ("incomplete", "failed") and result.exit_code in (1, 2)


def test_every_case_skipped_means_nothing_ran_exit_5() -> None:
    sk = CaseResult("s", "skipped", reason_code="skipped", reason="x", n_planned=1)
    assert compute_exit_code((sk, sk), (), interrupted=False, incomplete_policy="fail") == 5
    ok = CaseResult("p", "passed", n_planned=1)
    assert compute_exit_code((sk, ok), (), interrupted=False, incomplete_policy="fail") == 0


def test_fail_fast_skips_do_not_turn_a_failure_green() -> None:
    f = CaseResult("f", "failed", reason_code="below_threshold", reason="x", n_planned=1)
    sk = CaseResult("s", "skipped", reason_code="fail_fast", reason="x", n_planned=1)
    assert compute_exit_code((f, sk), (), interrupted=False, incomplete_policy="fail") == 1


def test_only_repeat_replay_of_a_passing_repeat_is_passed() -> None:
    r = run_suite(
        [refund_case("m", repeats=5)],
        agent_factory=make_good_agent,
        config=SuiteConfig(only_repeat=1),
    )
    assert r.cases[0].status == "passed" and r.exit_code == 0


def _split_part(case_id: str, status: str = "passed", only: int | None = None) -> Any:
    cfg = SuiteConfig(only_repeat=only, keep_reports="all") if only is not None else SuiteConfig()
    return run_suite(
        [refund_case(case_id, repeats=1)], agent_factory=make_good_agent, config=cfg
    ).cases[0]


def test_merge_refuses_parts_that_are_not_split_repeats() -> None:
    import pytest

    a, b = _split_part("same"), _split_part("same")  # two whole results with one id
    with pytest.raises(ValueError, match="duplicate case id"):
        merge_repeat_results([a, b])
    assert dataclasses.is_dataclass(a)


def test_a_relative_dataset_cannot_escape_the_case_file_directory(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("yaml")
    from disseqt_agent_testing.runner import load_cases

    (tmp_path / "secret.csv").write_text("name\nleaked\n")
    sub = tmp_path / "cases"
    sub.mkdir()
    (sub / "rows.csv").write_text("name\nok\n")
    head = 'version: 1\ncases:\n  - id: "t_${name}"\n    input: x\n    dataset: '

    def load(ref: str) -> Any:
        f = sub / "c.agent.yaml"
        f.write_text(head + ref + "\n")
        return load_cases(f)

    assert [c.case.id for c in load("rows.csv").cases] == ["t_ok"]
    escaped = load("../secret.csv")
    assert (
        not escaped.cases
        and "must stay inside the case file's directory" in escaped.errors[0].message
    )
    assert [c.case.id for c in load(str(tmp_path / "secret.csv")).cases] == [
        "t_leaked"
    ]  # explicit absolute
    link = sub / "link.csv"
    link.symlink_to(tmp_path / "secret.csv")
    assert not load("link.csv").cases  # a symlink out of the tree is an escape too


def _slow(delay: float) -> Any:
    return lambda tools: make_slow_agent(tools, delay)


def test_a_timeout_on_the_selected_repeat_is_incomplete_not_an_only_repeat_skip() -> None:
    cfg = SuiteConfig(max_workers=1, suite_timeout_s=0.25, only_repeat=1)
    cases = [refund_case(f"c{i}", repeats=2) for i in range(5)]
    result = run_suite(cases, agent_factory=_slow(0.2), config=cfg)
    cut = [c for c in result.cases if c.status != "passed"]
    assert cut and all((c.status, c.reason_code) == ("incomplete", "suite_timeout") for c in cut), [
        (c.case_id, c.status, c.reason_code) for c in result.cases
    ]
    assert result.exit_code == 2


def test_fail_fast_on_the_selected_repeat_is_still_a_deliberate_skip() -> None:
    from .._support.runner_agents import make_wrong_agent

    cfg = SuiteConfig(max_workers=1, fail_fast=True, only_repeat=1)
    cases = [refund_case(f"c{i}", repeats=2, pass_k=1) for i in range(3)]
    result = run_suite(cases, agent_factory=make_wrong_agent, config=cfg)
    later = [c for c in result.cases if c.status != "failed"]
    assert later and all((c.status, c.reason_code) == ("skipped", "fail_fast") for c in later)
    assert result.exit_code == 1


def test_a_suite_deadline_that_passes_before_planning_is_incomplete() -> None:
    cfg = SuiteConfig(suite_timeout_s=1e-9)
    result = run_suite([refund_case("late")], agent_factory=make_good_agent, config=cfg)
    case = result.cases[0]
    assert (case.status, case.reason_code) == ("incomplete", "suite_timeout")
    assert result.exit_code == 2


def test_a_partly_cut_off_case_exits_2_under_the_skip_policy() -> None:
    for policy in ("skip", "fail"):
        cfg = SuiteConfig(
            max_workers=1, case_timeout_s=0.25, run_timeout_s=5, incomplete_policy=policy
        )
        result = run_suite([refund_case("part", repeats=4)], agent_factory=_slow(0.15), config=cfg)
        case = result.cases[0]
        reasons = [r.not_run_reason for r in case.repeats]
        assert any(r.status == "passed" for r in case.repeats) and "case_timeout" in reasons
        assert (case.status, case.reason_code) == ("incomplete", "case_timeout")
        assert "case_timeout" in case.reason and result.exit_code == 2


def test_an_undecided_case_with_no_cut_off_keeps_its_code_and_the_skip_policy() -> None:
    from disseqt_agent_testing.models import Decision, Expect

    from .._support.runner_agents import refund_expect

    expect = Expect(
        decision=Decision.CALL, tools=refund_expect().tools, order="partial", max_tokens=100
    )
    result = run_suite(
        [refund_case("u", repeats=2, expect=expect)],
        agent_factory=make_good_agent,
        config=SuiteConfig(incomplete_policy="skip"),
    )
    case = result.cases[0]
    assert (case.status, case.reason_code) == ("incomplete", "undecided") and result.exit_code == 0


def _cut_part(part: CaseResult, index: int, reason: str) -> CaseResult:
    from disseqt_agent_testing.runner import RepeatRecord

    repeats = tuple(
        RepeatRecord(r.index, r.seed, "not_run", not_run_reason=reason) if r.index == index else r
        for r in part.repeats
    )
    return dataclasses.replace(
        part, status="incomplete", reason_code=reason, reason="x", repeats=repeats
    )


def test_merging_keeps_the_cut_off_reason_of_a_split_part() -> None:
    case = refund_case("m", repeats=2)
    parts = [
        run_suite(
            [case],
            agent_factory=make_good_agent,
            config=SuiteConfig(only_repeat=i, keep_reports="all"),
        ).cases[0]
        for i in range(2)
    ]
    merged = merge_repeat_results([parts[0], _cut_part(parts[1], 1, "suite_timeout")])
    assert [r.not_run_reason for r in merged.repeats] == [None, "suite_timeout"]
    assert (merged.status, merged.reason_code) == ("incomplete", "suite_timeout")
    assert "suite_timeout" in merged.reason
    assert compute_exit_code((merged,), (), interrupted=False, incomplete_policy="skip") == 2
    nothing_ran = merge_repeat_results(
        [_cut_part(parts[0], 0, "case_timeout"), _cut_part(parts[1], 1, "case_timeout")]
    )
    assert (nothing_ran.status, nothing_ran.reason_code) == ("incomplete", "case_timeout")
    assert compute_exit_code((nothing_ran,), (), interrupted=False, incomplete_policy="skip") == 2


def test_warnings_as_errors_never_abort_a_run(capsys: Any) -> None:
    import warnings

    def boom(event: Any) -> None:
        raise RuntimeError("boom")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = run_suite(
            [refund_case("w", repeats=4, pass_k=4)],
            agent_factory=make_good_agent,
            config=SuiteConfig(repeats=2),
            on_event=boom,
        )
    assert result.cases[0].status == "passed" and result.exit_code == 0
    err = capsys.readouterr().err
    assert "an on_event callback raised RuntimeError: boom" in err and "pass_k=4" in err


def test_the_async_engine_survives_warnings_as_errors() -> None:
    import asyncio
    import warnings

    from disseqt_agent_testing.runner import arun_suite

    def boom(event: Any) -> None:
        raise RuntimeError("boom")

    async def go() -> Any:
        return await arun_suite([refund_case("aw")], agent_factory=make_good_agent, on_event=boom)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert asyncio.run(go()).exit_code == 0


def test_a_filter_applies_to_an_already_loaded_suite() -> None:
    import asyncio

    from disseqt_agent_testing.runner import CaseFilter, arun_suite, load_cases

    loaded = load_cases([refund_case("a"), refund_case("b")])
    only_a = CaseFilter(ids=("a",))
    result = run_suite(loaded, agent_factory=make_good_agent, filter=only_a)
    assert [c.case_id for c in result.cases] == ["a"]
    again = asyncio.run(arun_suite(loaded, agent_factory=make_good_agent, filter=only_a))
    assert [c.case_id for c in again.cases] == ["a"]
    assert len(run_suite(loaded, agent_factory=make_good_agent).cases) == 2


def _stuck_factory(release: Any) -> Any:
    return lambda tools: lambda text: release.wait(10)


def test_a_case_timeout_ends_the_run_without_waiting_for_the_stuck_unit(
    monkeypatch: Any,
) -> None:
    import threading
    import time

    from disseqt_agent_testing.runner import engine

    monkeypatch.setattr(engine, "GRACE_S", 0.1)
    release = threading.Event()
    cfg = SuiteConfig(max_workers=2, case_timeout_s=0.1, run_timeout_s=4)
    start = time.monotonic()
    try:
        result = run_suite(
            [refund_case("stuck", repeats=2)], agent_factory=_stuck_factory(release), config=cfg
        )
        elapsed = time.monotonic() - start
    finally:
        release.set()
    case = result.cases[0]
    assert elapsed < 2.0, elapsed
    assert (case.status, case.reason_code) == ("incomplete", "case_timeout")
    assert result.exit_code == 2
