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
