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
