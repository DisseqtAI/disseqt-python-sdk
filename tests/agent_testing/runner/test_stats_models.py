"""Statistics, the bounds rule, summarize_repeats, seeds and the result models (ST-, SC-)."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.assertions import MetricResult, evaluate
from disseqt_agent_testing.assertions.report import DecisionReport, RepeatReport
from disseqt_agent_testing.models import Expect, ToolExpect
from disseqt_agent_testing.runner import (
    CaseResult,
    SuiteConfig,
    SuiteConfigError,
    SuiteResult,
    decide,
    derive_seed,
    failure_codes_of,
    pass_at_1,
    pass_at_k,
    pass_hat_k,
    summarize_repeats,
    wilson_95,
)
from disseqt_agent_testing.runner.models import RunBlock, compute_exit_code

from ..assertions.helpers import run_of, step


def r6(x: Any) -> Any:
    if x is None:
        return None
    return tuple(round(v, 6) for v in x) if isinstance(x, tuple) else round(x, 6)


def test_st01_st03_wilson_intervals() -> None:
    assert r6(wilson_95(3, 5)) == (0.230724, 0.882379)
    assert r6(wilson_95(5, 5)) == (0.565518, 1.0)
    assert r6(wilson_95(0, 5)) == (0.0, 0.434482)
    assert wilson_95(0, 0) is None
    assert r6(wilson_95(8, 10)) == (0.490162, 0.943318)


def test_st04_st05_pass_hat_k() -> None:
    assert [r6(pass_hat_k(*a)) for a in [(3, 5, 3), (4, 5, 3), (8, 10, 3), (3, 5, 5)]] == [
        0.1,
        0.4,
        0.466667,
        0.0,
    ]
    assert pass_hat_k(4, 4, 5) is None  # fewer decided repeats than k is never a free 1.0


def test_st06_pass_at_k_and_pass_at_1() -> None:
    assert [r6(pass_at_k(*a)) for a in [(1, 5, 2), (2, 10, 3), (3, 5, 3)]] == [0.4, 0.533333, 1.0]
    assert pass_at_k(1, 2, 3) is None and pass_at_1(0, 0) is None and pass_at_1(3, 4) == 0.75


@pytest.mark.parametrize(
    ("args", "want"),
    [
        ((5, 0, 0, 5, 1.0), "passed"),  # ST-07 all pass
        ((4, 1, 0, 5, 1.0), "failed"),  # one fail
        ((4, 0, 1, 5, 1.0), "undecided"),  # one incomplete
        ((3, 1, 1, 5, 1.0), "failed"),
        ((4, 1, 0, 3, 0.4), "passed"),  # an exact tie passes
        ((4, 0, 1, 3, 1.0), "undecided"),  # the worst case is 0.4
        ((0, 0, 5, 1, 1.0), "undecided"),  # zero decided can never pass
        ((0, 0, 0, 1, 1.0), "undecided"),
        ((3, 0, 0, 3, 0.8), "passed"),
        ((2, 1, 0, 3, 0.8), "failed"),
    ],
)
def test_st07_st12_the_bounds_rule(args: tuple[int, int, int, int, float], want: str) -> None:
    assert decide(*args) == want


def test_a_decimal_threshold_means_exactly_that_fraction() -> None:
    assert (
        decide(4, 1, 0, 3, 0.4) == "passed"
    )  # C(4,3)/C(5,3) = 4/10 is exactly 0.4, no float noise
    assert decide(7, 3, 0, 3, 0.29166666666666663) in (
        "passed",
        "failed",
    )  # no exception on ugly floats


def fake_report(outcome: str, label: str = "call", codes: tuple[str, ...] = ()) -> DecisionReport:
    checks = tuple(
        MetricResult("tool_selection", status="ok", value=0.0, passed=False, reason_code=c)
        for c in codes
    )
    return DecisionReport(
        "c", outcome, harness={"status": "finished", "violations": []}, decisions=({"turn": 0, "label": label},),
        checks=checks,
    )  # fmt: skip


def test_st13_summarize_matches_the_published_numbers() -> None:
    reports = [fake_report(o) for o in "passed passed failed passed failed".split()]
    rr = summarize_repeats(reports, k=5, threshold=1.0)
    assert (rr.n, rr.n_passed, rr.n_failed, rr.outcome, rr.worst) == (5, 3, 2, "failed", "2")
    assert r6(rr.wilson_95) == (0.230724, 0.882379) and rr.pass_at_1 == 0.6 and rr.pass_hat_k == 0.0


def test_st14_incomplete_and_not_run_repeats_are_undecided() -> None:
    pi = [fake_report("passed"), fake_report("incomplete"), fake_report("passed")]
    rr = summarize_repeats(pi, k=3, threshold=1.0)
    assert rr.outcome == "incomplete" and rr.n_incomplete == 1 and rr.pass_hat_k is None
    rr = summarize_repeats([fake_report("passed"), None, None], k=1, threshold=1.0, n_planned=3)
    assert rr.outcome == "incomplete" and rr.n == 1  # a not-run repeat is never a pass


def test_st15_st16_histogram_and_failure_code_accounting() -> None:
    reports = [
        fake_report("failed", "call", ("tool.missing",)),
        fake_report("failed", "ask", ("tool.missing", "decision.mismatch")),
        fake_report("passed", "call"),
    ]
    rr = summarize_repeats(reports, k=1, threshold=0.3)
    assert dict(rr.decision_histogram) == {"ask": 1, "call": 2}
    assert dict(rr.failure_codes) == {"decision.mismatch": 1, "tool.missing": 2}


def test_failure_codes_separate_agent_problems_from_instrument_problems() -> None:
    run = run_of(step(1, "a"))
    rep = evaluate(run, Expect(tools=(ToolExpect("zzz"),), max_llm_calls=1))
    codes = failure_codes_of(rep)
    assert "tool.missing" in codes and "unavailable:data_unavailable" in codes
    errored = DecisionReport(
        "c", "incomplete", harness={"status": "crashed", "violations": [{"code": "x"}]},
        checks=(MetricResult("m", status="error", reason_code="judge_failed"),),
    )  # fmt: skip
    assert failure_codes_of(errored) == [
        "error:judge_failed",
        "harness.not_finished",
        "harness.violation",
    ]


def test_derive_seed_is_stable_and_spread() -> None:
    assert derive_seed(0, "refund_happy", 0) == 1726685483762639142  # pinned: part of the contract
    assert derive_seed(0, "a", 1) != derive_seed(0, "b", 1)  # the case id is part of the input
    assert derive_seed(0, "a", 1) != derive_seed(0, "a", 2) != derive_seed(1, "a", 2)
    assert 0 <= derive_seed(5, "x", 3) < 2**63
    code = "from disseqt_agent_testing.runner import derive_seed; print(derive_seed(3, 'case', 7))"
    outs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": h, "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src")},
        ).stdout.strip()
        for h in ("0", "1", "random")
    }  # fmt: skip
    assert len(outs) == 1 and outs == {str(derive_seed(3, "case", 7))}


def test_sc_suite_result_round_trips_with_no_nan() -> None:
    from ..runner.test_engine import run_refund

    result = run_refund(3)
    data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    clone = SuiteResult.from_dict(data)
    assert clone.canonical_json() == result.canonical_json()
    data["from_the_future"] = 1
    assert SuiteResult.from_dict(data).canonical_json() == result.canonical_json()
    assert CaseResult.from_dict(result.cases[0].to_dict()).to_dict() == result.cases[0].to_dict()


def test_canonical_json_drops_volatile_fields_only() -> None:
    from ..runner.test_engine import run_refund

    a, b = run_refund(2), run_refund(2)
    assert a.suite_id != b.suite_id and a.canonical_json() == b.canonical_json()
    text = a.canonical_json()
    assert "suite_id" not in text and "created_at" not in text and "trace_id" not in text
    assert "case_digest" in text and "suite_digest" in text


def test_suite_write_is_atomic_and_valid_json(tmp_path: Path) -> None:
    from ..runner.test_engine import run_refund

    target = tmp_path / "out" / "disseqt.json"
    run_refund(1).write(target)
    assert json.loads(target.read_text())["schema_version"] == "1"
    assert not [p for p in target.parent.iterdir() if p.suffix == ".tmp"]


def test_suite_config_validation() -> None:
    for bad in ({"max_workers": 0}, {"repeats": 0}, {"run_timeout_s": 0}, {"incomplete_policy": "maybe"},
                {"judge_missing": "x"}, {"keep_reports": "x"}, {"only_repeat": -1}):  # fmt: skip
        with pytest.raises(SuiteConfigError):
            SuiteConfig(**bad)
    with pytest.raises(SuiteConfigError, match="DSQ-U03"):
        SuiteConfig(early_stop=True, repeat_mode="split")
    assert SuiteConfig.from_dict(SuiteConfig(seed=4).to_dict()).seed == 4


def test_run_block_decodes_known_keys_and_rejects_unknown() -> None:
    block = RunBlock.from_dict({"agent": "m:f", "timeout_s": 5, "requires": ["yaml"], "skip": "no"})
    assert (block.agent, block.timeout_s, block.requires, block.skip) == ("m:f", 5, ("yaml",), "no")
    with pytest.raises(ValueError, match="unknown field"):
        RunBlock.from_dict({"agnet": "typo"})
    with pytest.raises(ValueError, match="timeout_s"):
        RunBlock(timeout_s=0)


def test_exit_code_precedence() -> None:
    def case(status: str, reason: str | None = None) -> CaseResult:
        return CaseResult("c", status, reason_code=reason)

    from disseqt_agent_testing.runner import CollectionErrorRecord

    err = (CollectionErrorRecord("f", "boom"),)
    assert (
        compute_exit_code((case("passed"),), (), interrupted=False, incomplete_policy="fail") == 0
    )
    assert compute_exit_code((), (), interrupted=False, incomplete_policy="fail") == 5
    assert (
        compute_exit_code((case("failed"),), (), interrupted=False, incomplete_policy="fail") == 1
    )
    assert (
        compute_exit_code((case("incomplete"),), (), interrupted=False, incomplete_policy="fail")
        == 2
    )
    assert (
        compute_exit_code((case("incomplete"),), (), interrupted=False, incomplete_policy="skip")
        == 0
    )
    assert (
        compute_exit_code((case("failed"),), err, interrupted=False, incomplete_policy="fail") == 3
    )
    assert (
        compute_exit_code(
            (case("error", "harness_config"),), (), interrupted=False, incomplete_policy="fail"
        )
        == 3
    )
    assert (
        compute_exit_code(
            (case("error", "agent_setup_failed"),), (), interrupted=False, incomplete_policy="fail"
        )
        == 1
    )
    assert (
        compute_exit_code(
            (case("error", "internal"), case("failed")),
            err,
            interrupted=False,
            incomplete_policy="fail",
        )
        == 4
    )
    assert (
        compute_exit_code(
            (case("error", "internal"),), err, interrupted=True, incomplete_policy="fail"
        )
        == 130
    )


def test_repeat_report_serializes() -> None:
    rr = RepeatReport(
        5,
        n_passed=3,
        n_failed=2,
        k=5,
        threshold=1.0,
        pass_at_1=0.6,
        wilson_95=(0.2307, 0.8824),
        outcome="failed",
        worst="2",
    )
    assert RepeatReport.from_dict(rr.to_dict()).to_dict() == rr.to_dict()
