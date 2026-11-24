"""The pytest plugin through pytester (PL-)."""

import pytest

pytest.importorskip("yaml")  # YAML case files need the [yaml] extra
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.runner.schema import suite_result_schema

from .._support.schema_check import validate
from .conftest import (  # type: ignore[import-not-found]
    GOOD_EXPECT,
    case,
    casefile,
    subprocess_pytest,
)


def write(pt: Any, name: str, text: str) -> Path:
    path = pt.pytester.path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def out(result: Any) -> str:
    return result.stdout.str()


def test_pl01_passing_yaml_cases_have_file_scoped_node_ids(pt: Any) -> None:
    write(pt, "cases/refund.agent.yaml", casefile(case("refund_happy", repeats=3)))
    result = pt("-v")
    result.assert_outcomes(passed=1)
    assert "cases/refund.agent.yaml::refund_happy PASSED" in out(result)
    assert "disseqt: 1 agent cases, seed=0" in out(result)


def test_pl02_a_failing_case_renders_the_layout_without_a_traceback(pt: Any) -> None:
    write(pt, "refund.agent.yaml", casefile(case("flaky", "make_patterned_agent", repeats=5)))
    result = pt()
    result.assert_outcomes(failed=1)
    text = out(result)
    assert "agent case 'flaky' FAILED (refund.agent.yaml:3)" in text
    assert (
        "repeats   ✓✓✗✓✗   c=3/5 decided  pass@1=0.60  pass^5=0.00 < threshold 1.00  Wilson95 [0.23, 0.88]"
        in text
    )
    assert "decisions call×5" in text and "codes     ungrounded_value×2" in text
    assert (
        "worst     repeat #2" in text
        and "replay    pytest 'refund.agent.yaml::flaky' --disseqt-only-repeat 2 --disseqt-seed 0"
        in text
    )
    assert "Traceback" not in text and "AgentCaseFailed" not in text


INCOMPLETE = "    expect:\n      decision: call\n      tools: [{name: get_order}, {name: issue_refund}]\n      max_tokens: 100\n"


def test_pl03_pl04_incomplete_fails_by_default_and_skips_on_request(pt: Any) -> None:
    write(pt, "inc.agent.yaml", casefile(case("undecidable", repeats=2, expect=INCOMPLETE)))
    result = pt("-v")
    assert "INCOMPLETE" in out(result) and "incomplete: case 'undecidable'" in out(result)
    assert result.ret == 1
    skipped = pt("-v", "--disseqt-incomplete=skip")
    assert skipped.ret == 0 and "1 incomplete" in out(skipped)


def test_pl05_one_broken_file_does_not_stop_a_valid_one(pt: Any) -> None:
    write(pt, "good.agent.yaml", casefile(case("good_case")))
    write(pt, "bad.agent.yaml", "version: 1\ncases:\n  - id: 'bad id'\n    input: x\n")
    result = pt("-v", "--continue-on-collection-errors")
    assert "good.agent.yaml::good_case PASSED" in out(result)
    assert "ERROR collecting bad.agent.yaml" in out(
        result
    ) and "case id 'bad id' is invalid" in out(result)
    assert result.ret == 1


def test_pl06_no_factory_is_a_setup_error_with_the_dsq_message(pt: Any) -> None:
    write(pt, "nf.agent.yaml", casefile(case("nofactory", agent=None)))
    result = pt("-v")
    result.assert_outcomes(errors=1)
    assert "case 'nofactory' has no agent factory: set 'run: {agent: module:factory}'" in out(
        result
    )


def test_pl07_a_conftest_hook_supplies_the_factory_scoped_by_directory(pt: Any) -> None:
    write(
        pt,
        "a/conftest.py",
        "import tests.agent_testing._support.runner_agents as m\n\ndef pytest_disseqt_agent_factory(case, config):\n    return m.make_good_agent\n",
    )
    write(pt, "a/x.agent.yaml", casefile(case("in_a", agent=None)))
    write(pt, "b/y.agent.yaml", casefile(case("in_b", agent=None)))
    result = pt("-v")
    assert "a/x.agent.yaml::in_a PASSED" in out(result) and "b/y.agent.yaml::in_b ERROR" in out(
        result
    )


def test_pl08_pl09_status_letters_and_the_failures_section(pt: Any) -> None:
    write(pt, "m.agent.yaml", casefile(case("ok_case"), case("bad_case", "make_wrong_agent")))
    result = pt("-v")
    text = out(result)
    assert "m.agent.yaml::ok_case PASSED" in text and "m.agent.yaml::bad_case FAILED" in text
    assert "= FAILURES =" in text and "disseqt agent results" in text
    assert text.index("FAILED     bad_case") < text.index("PASSED     ok_case")  # worst first


def test_pl10_a_missing_judge_skips_or_errors(pt: Any) -> None:
    judged = casefile(case("needs_judge", extra="    use_judge: [decision]"))
    write(pt, "j.agent.yaml", judged)
    assert "1 skipped" in out(pt("-rs")) and "judge unavailable" in out(pt("-rs"))
    strict = pt("--disseqt-require-judge")
    strict.assert_outcomes(errors=1)
    assert "no judge is configured" in out(strict)


def test_pl11_selection_by_k_m_and_node_id(pt: Any) -> None:
    write(
        pt,
        "s.agent.yaml",
        casefile(case("alpha", tags="smoke"), case("beta"), case("gamma", tags="smoke")),
    )
    assert pt("-k", "alpha").parseoutcomes()["passed"] == 1
    assert pt("-m", "smoke").parseoutcomes()["passed"] == 2
    assert pt("s.agent.yaml::beta").parseoutcomes()["passed"] == 1


def test_pl12_pl13_tag_and_id_filters_deselect_only_agent_items(pt: Any) -> None:
    write(
        pt,
        "s.agent.yaml",
        casefile(case("alpha", tags="smoke"), case("beta", tags="slow"), case("gamma")),
    )
    write(pt, "test_plain.py", "def test_plain():\n    assert True\n")
    r = pt("--disseqt-tags", "smoke")
    assert r.parseoutcomes()["passed"] == 2 and r.parseoutcomes()["deselected"] == 2  # beta, gamma
    r = pt("--disseqt-tags", "!slow", "--disseqt-case", "g*")
    assert r.parseoutcomes()["passed"] == 2  # gamma + the untouched plain test
    r = pt("--disseqt-tags", "smoke,slow", "--disseqt-tags", "!smoke")
    assert r.parseoutcomes()["passed"] == 2 and r.parseoutcomes()["deselected"] == 2


def test_pl14_strict_markers_and_pytest_9_strict_with_tags(pt: Any) -> None:
    write(pt, "t.agent.yaml", casefile(case("tagged", tags="refund, nightly")))
    assert pt("--strict-markers").parseoutcomes()["passed"] == 1
    pt.pytester.makeini("[pytest]\nstrict_markers = true\n")
    assert pt().parseoutcomes()["passed"] == 1


def test_pl15_run_skip_and_run_xfail(pt: Any) -> None:
    write(
        pt,
        "r.agent.yaml",
        casefile(
            case("skipped_one", extra='    run: {skip: "needs sandbox"}', agent=None),
            case("xfailing", "make_wrong_agent", extra=""),
        ),
    )
    r = pt("-rs")
    assert (
        "needs sandbox" in out(r)
        and r.parseoutcomes()["skipped"] == 1
        and r.parseoutcomes()["failed"] == 1
    )
    xf = casefile(
        case("expected_fail", "make_wrong_agent").replace(
            '    run: {agent: "tests.agent_testing._support.runner_agents:make_wrong_agent"}',
            '    run: {agent: "tests.agent_testing._support.runner_agents:make_wrong_agent", xfail: "known bug"}',
        )
    )
    write(pt, "x.agent.yaml", xf)
    assert "1 xfailed" in out(pt("x.agent.yaml"))


def test_pl16_a_force_override_below_pass_k_clamps_with_a_warning(pt: Any) -> None:
    write(pt, "p.agent.yaml", casefile(case("clamped", extra="    repeats: 4\n    pass_k: 4")))
    r = pt("--disseqt-repeats", "2", "-W", "default")
    assert r.parseoutcomes()[
        "passed"
    ] == 1 and "--disseqt-repeats 2 is smaller than pass_k=4; using k=2" in out(r)


def test_pl17_python_cases_with_the_marker_and_parametrize_cases_render_the_same_layout(
    pt: Any,
) -> None:
    write(
        pt,
        "test_py.py",
        """
import pytest
from disseqt_agent_testing.runner import CaseSpec
from disseqt_agent_testing.pytest_plugin import parametrize_cases
from tests.agent_testing._support.runner_agents import make_patterned_agent, refund_case

@pytest.mark.agent(repeats=5)
def test_marker(disseqt_run_case):
    disseqt_run_case(refund_case("py_flaky"), agent_factory=make_patterned_agent)

@parametrize_cases([refund_case("param_one", repeats=2)])
def test_param(agent_case, disseqt_run_case):
    disseqt_run_case(agent_case, agent_factory=make_patterned_agent)
""",
    )
    r = pt("-v")
    text = out(r)
    assert r.parseoutcomes()["failed"] == 1 and r.parseoutcomes()["passed"] == 1
    assert "repeats   ✓✓✗✓✗   c=3/5 decided  pass@1=0.60  pass^5=0.00 < threshold 1.00" in text
    assert "test_param[param_one]" in text


def test_pl18_two_run_case_calls_in_one_test(pt: Any) -> None:
    write(
        pt,
        "test_two.py",
        """
from tests.agent_testing._support.runner_agents import make_good_agent, refund_case

def test_two(disseqt_run_case):
    a = disseqt_run_case(refund_case("one", repeats=2), agent_factory=make_good_agent)
    b = disseqt_run_case(refund_case("two", repeats=3), agent_factory=make_good_agent)
    assert (a.n_planned, b.n_planned) == (2, 3)
""",
    )
    pt().assert_outcomes(passed=1)


def test_pl19_only_repeat_replays_one_repeat_with_its_derived_seed(pt: Any) -> None:
    write(pt, "o.agent.yaml", casefile(case("replayed", "make_patterned_agent", repeats=5)))
    r = pt("--disseqt-only-repeat", "2", "--disseqt-seed", "7", "--disseqt-output", "o.json")
    assert r.ret == 1  # repeat 2 is the failing one
    data = json.loads((pt.pytester.path / "o.json").read_text())
    reps = data["cases"][0]["repeats"]
    from disseqt_agent_testing.runner import derive_seed

    assert [x["status"] for x in reps] == ["not_run", "not_run", "failed", "not_run", "not_run"]
    assert reps[2]["seed"] == derive_seed(7, "replayed", 2)
    bad = pt("--disseqt-only-repeat", "9")
    assert (
        "--disseqt-only-repeat 9 is out of range for case 'replayed' with 5 repeats" in out(bad)
        or bad.ret != 0
    )


def test_pl20_invalid_option_values_exit_with_usage_status(
    pt: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(pt, "u.agent.yaml", casefile(case("c")))
    r = pt("--disseqt-run-timeout", "nope")
    assert (
        r.ret == 4
        and "invalid value 'nope' for --disseqt-run-timeout (CLI): seconds > 0"
        in "\n".join(r.stderr.lines + r.stdout.lines)
    )
    monkeypatch.setenv("DISSEQT_EARLY_STOP", "maybe")
    r = pt()
    assert r.ret == 4 and "DISSEQT_EARLY_STOP" in "\n".join(r.stderr.lines + r.stdout.lines)
    monkeypatch.delenv("DISSEQT_EARLY_STOP")
    assert pt("--disseqt-workers", "0").ret == 4
    both = pt("--disseqt-early-stop", "--disseqt-split-repeats")
    assert both.ret == 4 and "cannot be combined" in "\n".join(
        both.stderr.lines + both.stdout.lines
    )


def test_option_precedence_cli_over_env_over_ini(pt: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    write(pt, "p.agent.yaml", casefile(case("c")))
    pt.pytester.makeini("[pytest]\ndisseqt_seed = 11\n")
    assert "seed=11" in out(pt())
    monkeypatch.setenv("DISSEQT_SEED", "22")
    assert "seed=22" in out(pt())
    assert "seed=33" in out(pt("--disseqt-seed", "33"))


def test_pl23_junit_properties_are_compact_and_the_transient_one_is_gone(pt: Any) -> None:
    write(pt, "j.agent.yaml", casefile(case("junit_case", repeats=3)))
    r = pt("--junitxml", "j.xml")
    assert r.ret == 0
    root = ET.parse(pt.pytester.path / "j.xml").getroot()
    props = {p.get("name"): p.get("value") for p in root.iter("property")}
    assert props["disseqt.case_id"] == "junit_case" and props["disseqt.status"] == "passed"
    assert (
        props["disseqt.n"] == "3"
        and props["disseqt.pass_hat_k"] == "1.0"
        and "disseqt.seed" in props
    )
    assert "disseqt.case_result" not in props


def test_pl24_the_output_file_validates_against_the_committed_schema(pt: Any) -> None:
    write(pt, "o.agent.yaml", casefile(case("ok_case"), case("bad_case", "make_wrong_agent")))
    pt("--disseqt-output", "build/out.json")
    data = json.loads((pt.pytester.path / "build" / "out.json").read_text())
    schema = suite_result_schema()
    assert validate(data, schema, schema) == []
    assert (
        data["exit_code"] == 1 and data["totals"]["failed"] == 1 and data["schema_version"] == "1"
    )
    golden = Path(__file__).parent / "golden" / "suite_result_schema_v1.json"
    assert json.loads(golden.read_text()) == schema


def test_pl25_pl26_header_and_summary_text(pt: Any) -> None:
    write(pt, "g.agent.yaml", casefile(case("good", repeats=2)))
    text = out(pt("--disseqt-run-label", "nightly"))
    assert "disseqt: seed=0 workers=8" in text  # the header appears because an option was set
    assert "= disseqt agent results =" in text
    assert "PASSED     good                             ✓✓          pass@1=1.00 pass^2=1.00" in text
    assert (
        "1 cases: 1 passed, 0 failed, 0 incomplete, 0 skipped, 0 error; repeats run 2, not run 0"
        in text
    )


def test_pl27_pl28_pl29_warnings_for_rerun_plugins_and_timeouts(pt: Any) -> None:
    write(pt, "w.agent.yaml", casefile(case("c", repeats=2)))
    write(
        pt,
        "conftest.py",
        """
def pytest_configure(config):
    for name in ("rerunfailures", "repeat"):
        config.pluginmanager.register(object(), name)
""",
    )
    r = pt("-W", "default")
    text = out(r)
    assert "pytest-rerunfailures is active: reruns are disabled on agent items" in text
    assert "pytest-repeat is active: use --disseqt-repeats" in text


def test_pl30_split_mode_aggregates_to_the_same_case_result(pt: Any) -> None:
    write(pt, "s.agent.yaml", casefile(case("flaky", "make_patterned_agent", repeats=5)))
    internal = pt("--disseqt-output", "internal.json")
    split = pt("-v", "--disseqt-split-repeats", "--disseqt-output", "split.json")
    assert "flaky[r0] PASSED" in out(split) and "flaky[r2] FAILED" in out(split)
    a = json.loads((pt.pytester.path / "internal.json").read_text())["cases"][0]
    b = json.loads((pt.pytester.path / "split.json").read_text())["cases"][0]
    assert a["status"] == b["status"] == "failed" and a["repeat_report"] == b["repeat_report"]
    assert [r["status"] for r in a["repeats"]] == [r["status"] for r in b["repeats"]]
    assert internal.ret == 1 and split.ret == 1


def test_pl31_the_plugin_is_silent_and_light_when_unused(tmp_path: Path) -> None:
    (tmp_path / "test_plain.py").write_text("def test_plain():\n    assert True\n")
    code = (
        "import sys, pytest\n"
        "rc = pytest.main(['-p', 'disseqt_agent_testing.pytest_plugin', '-p', 'no:cacheprovider', 'test_plain.py', '-q'])\n"
        "heavy = [m for m in sys.modules if m.startswith('disseqt_agent_testing.') and any(m.startswith(f'disseqt_agent_testing.{x}') for x in ('runner', 'harness', 'assertions', 'judges'))]\n"
        "print('HEAVY:', sorted(heavy))\n"
        "raise SystemExit(rc)\n"
    )
    import os
    import subprocess
    import sys

    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src"),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert (
        "disseqt"
        not in r.stdout.replace("HEAVY:", "").replace("disseqt_agent_testing", "x").lower()
    )
    assert "HEAVY: []" in r.stdout and "warning" not in r.stdout.lower()


def test_pl21_pl22_the_plugin_can_be_switched_off(tmp_path: Path) -> None:
    (tmp_path / "x.agent.yaml").write_text("version: 1\ncases: []\n")
    (tmp_path / "test_a.py").write_text("def test_a():\n    pass\n")
    on = subprocess_pytest(
        tmp_path,
        "-p",
        "disseqt_agent_testing.pytest_plugin",
        "-q",
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    assert on.returncode in (0, 5)
    off = subprocess_pytest(
        tmp_path, "-q", "--disseqt-seed", "1", env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    )
    assert off.returncode == 4 and "unrecognized arguments: --disseqt-seed" in off.stderr


_ = (GOOD_EXPECT,)


@pytest.mark.skipif(
    int(pytest.__version__.split(".")[0]) < 9, reason="native [tool.pytest] config needs pytest 9"
)
def test_cp02_native_toml_configuration_on_pytest_9(pt: Any) -> None:
    write(pt, "n.agent.yaml", casefile(case("native", repeats=3)))
    write(
        pt,
        "pyproject.toml",
        '[tool.pytest]\ndisseqt_seed = 11\ndisseqt_workers = 2\ndisseqt_case_globs = ["*.agent.yaml"]\n',
    )
    r = pt("--disseqt-output", "n.json")
    assert r.ret == 0, out(r)
    data = json.loads((pt.pytester.path / "n.json").read_text())
    assert data["seed"] == 11 and data["config"]["max_workers"] == 2


def test_the_case_result_hook_and_suite_result_hook_fire_on_the_controller(pt: Any) -> None:
    write(pt, "h.agent.yaml", casefile(case("hooked", repeats=2)))
    write(
        pt,
        "conftest.py",
        """
seen = []

def pytest_disseqt_case_result(result):
    seen.append(("case", result.case_id, result.status))

def pytest_disseqt_suite_result(result):
    seen.append(("suite", result.totals.cases, result.exit_code))

def pytest_sessionfinish(session):
    print("HOOKS:", seen)
""",
    )
    r = pt("-s")
    assert "HOOKS: [('case', 'hooked', 'passed'), ('suite', 1, 0)]" in out(
        r
    ) or "('case', 'hooked', 'passed')" in out(r)


def test_pytest_skip_inside_a_case_agent_is_a_skip_not_a_failure(pt: Any) -> None:
    write(
        pt,
        "test_skip.py",
        """
import pytest
from tests.agent_testing._support.runner_agents import refund_case

def test_skip(disseqt_run_case):
    def factory(tools):
        return lambda text: pytest.skip("agent wants out")
    with pytest.raises(pytest.skip.Exception):
        disseqt_run_case(refund_case("sk"), agent_factory=factory)
""",
    )
    pt().assert_outcomes(passed=1)
