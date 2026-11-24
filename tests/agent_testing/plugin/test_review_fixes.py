"""Review fix pass for the plugin: duplicate ids, inertness, ini coercion."""

import pytest

pytest.importorskip("yaml")
import json
from typing import Any

from .conftest import case, casefile  # type: ignore[import-not-found]
from .test_plugin import out, write


def both(r: Any) -> str:
    return "\n".join(r.stderr.lines + r.stdout.lines)


def test_two_tests_with_one_case_id_never_merge_into_a_green_result(pt: Any) -> None:
    write(
        pt,
        "test_dup.py",
        """
from tests.agent_testing._support.runner_agents import (
    make_good_agent, make_wrong_agent, refund_case)

def test_good(disseqt_run_case):
    disseqt_run_case(refund_case("same", repeats=2), agent_factory=make_good_agent)

def test_bad(disseqt_run_case):
    disseqt_run_case(refund_case("same", repeats=2), agent_factory=make_wrong_agent)
""",
    )
    r = pt("--disseqt-output", "o.json")
    assert r.ret != 0
    r.assert_outcomes(passed=1, failed=1)
    data = json.loads((pt.pytester.path / "o.json").read_text())
    by = {c["case_id"]: c["status"] for c in data["cases"]}
    assert by == {"same": "passed", "same#2": "failed"} or sorted(by.values()) == [
        "failed",
        "passed",
    ]
    assert "incomplete" not in by.values()
    assert any(
        e["code"] == "DSQ-C03" and "duplicate case id 'same'" in e["message"]
        for e in data["collection_errors"]
    )
    assert data["exit_code"] != 0


def test_split_repeat_items_of_one_case_still_merge(pt: Any) -> None:
    write(pt, "s.agent.yaml", casefile(case("one", "make_good_agent", repeats=3)))
    r = pt("--disseqt-split-repeats", "--disseqt-output", "o.json")
    r.assert_outcomes(passed=3)
    data = json.loads((pt.pytester.path / "o.json").read_text())
    assert [c["case_id"] for c in data["cases"]] == ["one"] and not data["collection_errors"]
    assert data["cases"][0]["status"] == "passed"
    assert out(r)


def test_an_unrelated_agent_json_file_is_ignored_silently(pt: Any) -> None:
    write(pt, "support.agent.json", '{"name": "bot"}')
    write(pt, "other.agent.yaml", "name: bot\nmodel: x\n")
    write(pt, "test_plain.py", "def test_plain():\n    assert True\n")
    r = pt()
    assert r.ret == 0, out(r)
    r.assert_outcomes(passed=1)
    assert "support.agent.json" not in out(r)


def test_a_real_case_file_with_a_syntax_error_is_still_reported(pt: Any) -> None:
    write(pt, "bad.agent.json", '{"version": 1, "cases": [')
    r = pt()
    assert r.ret != 0 and "bad.agent.json" in out(r)


def test_an_explicit_disseqt_option_reports_problems_in_matching_files(pt: Any) -> None:
    write(pt, "support.agent.json", '{"name": "bot"}')
    r = pt("--disseqt-repeats", "2")
    assert r.ret != 0 and "support.agent.json" in out(r)


def test_an_invalid_seed_env_does_not_abort_unrelated_runs(
    pt: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISSEQT_SEED", "abc")
    write(pt, "test_plain.py", "def test_plain():\n    assert True\n")
    r = pt()
    assert r.ret == 0, out(r)
    r.assert_outcomes(passed=1)


def test_an_invalid_seed_env_errors_when_the_plugin_is_used(
    pt: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISSEQT_SEED", "abc")
    write(pt, "ok.agent.yaml", casefile(case("c1", "make_good_agent")))
    r = pt()
    assert r.ret != 0 and "invalid value 'abc'" in both(r) and "DISSEQT_SEED" in both(r)
    write(
        pt,
        "test_fx.py",
        "def test_fx(disseqt_config):\n    pass\n",
    )
    (pt.pytester.path / "ok.agent.yaml").unlink()
    r = pt()
    assert r.ret != 0 and "DISSEQT_SEED" in both(r)


def test_an_invalid_seed_on_the_command_line_still_errors_at_once(pt: Any) -> None:
    write(pt, "test_plain.py", "def test_plain():\n    assert True\n")
    r = pt("--disseqt-seed", "abc")
    assert r.ret != 0 and "invalid value 'abc'" in "\n".join(r.stderr.lines + r.stdout.lines)


@pytest.mark.parametrize(
    ("raw", "text"),
    [(11, "11"), (2.5, "2.5"), (True, "true"), (False, "false"), ("random", "random"),
     (["*.a.yaml", "*.b.json"], "*.a.yaml *.b.json"), (None, None), ("", None), ([], None)],
)  # fmt: skip
def test_ini_values_of_any_scalar_type_become_text(raw: Any, text: Any) -> None:
    from disseqt_agent_testing.pytest_plugin.options import ini_text_from_raw

    assert ini_text_from_raw(raw) == text


def test_a_strict_ini_type_error_falls_back_to_the_raw_value() -> None:
    """pytest 9.2 raises UsageError (earlier: TypeError) when a TOML int meets a string option."""
    from disseqt_agent_testing.pytest_plugin.options import _ini_text

    class Strict:
        inicfg = {"disseqt_seed": 11, "disseqt_ascii": True}

        def getini(self, name: str) -> Any:
            raise pytest.UsageError(f"config option '{name}' expects a string, got int: 11")

    assert _ini_text(Strict(), "disseqt_seed") == "11"  # type: ignore[arg-type]
    assert _ini_text(Strict(), "disseqt_ascii") == "true"  # type: ignore[arg-type]
    assert _ini_text(Strict(), "disseqt_absent") is None  # type: ignore[arg-type]

    class Old(Strict):
        def getini(self, name: str) -> Any:
            raise TypeError("expects a string")

    assert _ini_text(Old(), "disseqt_seed") == "11"  # type: ignore[arg-type]


def _result(pt: Any, name: str = "o.json") -> Any:
    return json.loads((pt.pytester.path / name).read_text())


def _slow(case_id: str) -> str:
    return case(case_id, "make_slow_agent", repeats=3)


@pytest.mark.parametrize("policy", ["fail", "skip"])
def test_the_suite_timeout_spans_cases_in_pytest_mode(pt: Any, policy: str) -> None:
    write(pt, "s.agent.yaml", casefile(_slow("s1"), _slow("s2"), _slow("s3")))
    r = pt(
        "--disseqt-workers", "1", "--disseqt-suite-timeout", "0.45", "--disseqt-incomplete", policy,
        "--disseqt-output", "o.json",
    )  # fmt: skip
    data = _result(pt)
    last = next(c for c in data["cases"] if c["case_id"] == "s3")
    assert (last["status"], last["reason_code"]) == ("incomplete", "suite_timeout")
    assert all(rep["status"] == "not_run" for rep in last["repeats"])
    assert data["exit_code"] == 2 and r.ret != 0
    assert "suite timeout (0.45s)" in last["reason"]


def test_the_suite_timeout_spans_python_agent_tests(pt: Any) -> None:
    write(
        pt,
        "test_slow.py",
        """
from tests.agent_testing._support.runner_agents import make_slow_agent, refund_case

def test_a(disseqt_run_case):
    disseqt_run_case(refund_case("a", repeats=3), agent_factory=make_slow_agent)

def test_b(disseqt_run_case):
    disseqt_run_case(refund_case("b", repeats=3), agent_factory=make_slow_agent)
""",
    )
    r = pt("--disseqt-workers", "1", "--disseqt-suite-timeout", "0.2", "--disseqt-output", "o.json")
    data = _result(pt)
    by = {c["case_id"]: c for c in data["cases"]}
    assert (by["b"]["status"], by["b"]["reason_code"]) == ("incomplete", "suite_timeout")
    assert data["exit_code"] == 2 and r.ret != 0


def test_split_repeats_cut_by_the_suite_timeout_stay_one_incomplete_case(pt: Any) -> None:
    write(pt, "s.agent.yaml", casefile(_slow("s1"), _slow("s2")))
    r = pt(
        "--disseqt-split-repeats", "--disseqt-suite-timeout", "0.25", "--disseqt-incomplete", "skip",
        "--disseqt-output", "o.json",
    )  # fmt: skip
    data = _result(pt)
    assert [c["case_id"] for c in data["cases"]] == ["s1", "s2"]
    assert data["collection_errors"] == []
    last = data["cases"][1]
    assert (last["status"], last["reason_code"]) == ("incomplete", "suite_timeout")
    assert data["exit_code"] == 2 and r.ret != 0


def test_a_repeated_case_option_keeps_every_value(pt: Any) -> None:
    write(
        pt,
        "c.agent.yaml",
        casefile(case("ok1"), case("bad3", "make_wrong_agent"), case("other")),
    )
    for order in (("bad3", "ok1"), ("ok1", "bad3")):
        args = [a for cid in order for a in ("--disseqt-case", cid)]
        pt(*args).assert_outcomes(passed=1, failed=1, deselected=1)
    pt("--disseqt-case", "ok1,other", "--disseqt-case", "bad3").assert_outcomes(passed=2, failed=1)


def _xfail_case(case_id: str, agent: str, repeats: int | None = None) -> str:
    return case(case_id, agent, repeats=repeats).replace(
        f':{agent}"}}', f':{agent}", xfail: "known bug"}}'
    )


@pytest.mark.parametrize("split", [False, True])
def test_an_xfail_case_that_fails_is_an_expected_failure(pt: Any, split: bool) -> None:
    write(pt, "x.agent.yaml", casefile(_xfail_case("xf", "make_wrong_agent", repeats=2)))
    extra = ["--disseqt-split-repeats"] if split else []
    r = pt("--disseqt-output", "o.json", "-rx", *extra)
    assert r.ret == 0, out(r)
    r.assert_outcomes(xfailed=2 if split else 1)
    assert "known bug" in out(r)
    data = _result(pt)
    (only,) = data["cases"]
    assert (only["status"], only["reason_code"]) == ("xfailed", "expected_failure")
    assert data["exit_code"] == 0 and data["totals"]["xfailed"] == 1
    assert "XFAILED" in out(r) and "1 xfailed" in out(r)


@pytest.mark.parametrize("split", [False, True])
def test_an_xfail_case_that_passes_fails_the_run(pt: Any, split: bool) -> None:
    write(pt, "x.agent.yaml", casefile(_xfail_case("xp", "make_good_agent", repeats=2)))
    extra = ["--disseqt-split-repeats"] if split else []
    r = pt("--disseqt-output", "o.json", *extra)
    assert r.ret == 1, out(r)
    data = _result(pt)
    (only,) = data["cases"]
    assert (only["status"], only["reason_code"]) == ("failed", "unexpected_pass")
    assert "known bug" in only["reason"]
    assert data["exit_code"] == 1


def test_runxfail_reports_an_xfail_case_as_a_plain_failure(pt: Any) -> None:
    write(pt, "x.agent.yaml", casefile(_xfail_case("xf", "make_wrong_agent")))
    r = pt("--runxfail", "--disseqt-output", "o.json")
    r.assert_outcomes(failed=1)
    data = _result(pt)
    assert data["cases"][0]["status"] == "failed" and data["exit_code"] == 1


def test_a_bad_env_combination_does_not_break_a_project_without_agent_cases(
    pt: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISSEQT_EARLY_STOP", "1")
    monkeypatch.setenv("DISSEQT_SPLIT_REPEATS", "1")
    write(pt, "test_plain.py", "def test_ok():\n    assert True\n")
    r = pt()
    assert r.ret == 0, both(r)
    r.assert_outcomes(passed=1)
    write(pt, "c.agent.yaml", casefile(case("c1")))
    used = pt()
    assert used.ret == 4 and "cannot be combined" in both(used)
    explicit = pt("--disseqt-split-repeats", "test_plain.py")
    assert explicit.ret == 4 and "cannot be combined" in both(explicit)


def test_split_repeats_keeps_a_skipped_case_as_one_case(pt: Any) -> None:
    skipped = case("sk", repeats=3).replace('"}', '", skip: "later"}', 1)
    write(pt, "k.agent.yaml", casefile(skipped))
    plain = pt("--disseqt-output", "plain.json")
    split = pt("--disseqt-split-repeats", "--disseqt-output", "split.json")
    assert plain.ret == split.ret == 0
    for name in ("plain.json", "split.json"):
        data = _result(pt, name)
        assert [(c["case_id"], c["status"]) for c in data["cases"]] == [("sk", "skipped")]
        assert data["collection_errors"] == []
    assert _result(pt, "split.json")["exit_code"] == _result(pt, "plain.json")["exit_code"]


def test_split_repeats_keeps_a_setup_error_case_as_one_case(pt: Any) -> None:
    write(pt, "e.agent.yaml", casefile(case("nofactory", agent=None, repeats=3)))
    r = pt("--disseqt-split-repeats", "--disseqt-output", "o.json")
    r.assert_outcomes(errors=3)
    data = _result(pt)
    assert [(c["case_id"], c["reason_code"]) for c in data["cases"]] == [
        ("nofactory", "no_agent_factory")
    ]
    assert data["collection_errors"] == []


def test_split_repeats_still_reports_a_real_duplicate_id(pt: Any) -> None:
    write(
        pt,
        "test_dup.py",
        """
from tests.agent_testing._support.runner_agents import make_good_agent, refund_case

def test_one(disseqt_run_case):
    disseqt_run_case(refund_case("same"), agent_factory=make_good_agent)

def test_two(disseqt_run_case):
    disseqt_run_case(refund_case("same"), agent_factory=make_good_agent)
""",
    )
    pt("--disseqt-output", "o.json")
    data = _result(pt)
    assert [e["code"] for e in data["collection_errors"]] == ["DSQ-C03"]
    assert data["exit_code"] == 3


def test_an_output_file_is_written_even_when_no_agent_case_ran(pt: Any) -> None:
    write(pt, "test_plain.py", "def test_ok():\n    assert True\n")
    r = pt("--disseqt-output", "o.json")
    r.assert_outcomes(passed=1)
    data = _result(pt)
    assert data["cases"] == [] and data["exit_code"] == 5
