"""Edge behaviour: canonical JSON, Ref sources, rendering of unusual results, bad inputs."""

import math
from typing import Any

import pytest

from disseqt_agent_testing.assertions import (
    ExpectConfigError,
    MetricResult,
    aevaluate,
    assert_agent,
    evaluate,
)
from disseqt_agent_testing.assertions.assignment import assign
from disseqt_agent_testing.assertions.canonical import (
    args_hash,
    canonical_json,
    clip,
    normalize,
    path_str,
    round6,
)
from disseqt_agent_testing.assertions.config import EvalConfig, resolve_config
from disseqt_agent_testing.assertions.render import default_lines, render, resolve_max_lines
from disseqt_agent_testing.models import (
    Expect,
    FakeResponse,
    FakeSpec,
    Ref,
    ToolExpect,
)

from .helpers import run_of, step


def test_canonical_json_is_order_independent_and_rounds_floats() -> None:
    assert canonical_json({"b": 1, "a": [1.0000001, None]}) == '{"a":[1.0,null],"b":1}'
    assert args_hash({"x": 1, "y": 2}) == args_hash({"y": 2, "x": 1})
    assert args_hash({"x": 1}) != args_hash({"x": "1"})
    assert round6(-0.0) == 0.0 and str(round6(-0.0)) == "0.0" and round6(0.1234567) == 0.123457


def test_canonical_json_rejects_what_json_cannot_hold() -> None:
    with pytest.raises(ValueError):
        canonical_json({"a": math.nan})
    with pytest.raises(TypeError):
        normalize({"a": object()})


def test_paths_and_clipping() -> None:
    assert path_str(("items", 0, "city")) == "$.items[0].city" and path_str(()) == "$"
    long = "x" * 200
    assert clip(long).endswith("...(len=202)") and len(clip(long)) <= 80 + len("...(len=202)")
    assert clip("short") == "'short'" and clip({"a": 1}) == '{"a":1}'
    assert clip(object()).startswith("<object")
    assert "\\n" in clip("a\nb")


def test_bad_inputs_are_type_errors() -> None:
    with pytest.raises(TypeError, match="RunResult or AgentRun"):
        evaluate("not a run", Expect())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be an Expect"):
        evaluate(run_of(), "nope")  # type: ignore[arg-type]


def test_config_resolution_precedence_and_unknown_keys() -> None:
    base = EvalConfig(strict=False, lang="en")
    assert resolve_config(base, {"strict": True}).strict is True and base.strict is False
    assert resolve_config(base, {}) is base
    assert resolve_config(None, {"lang": "en"}).lang == "en"
    with pytest.raises(TypeError, match="unknown option"):
        resolve_config(None, {"nope": 1})
    for bad in (
        {"on_unavailable": "maybe"},
        {"min_pair_score": -1},
        {"lexicons": 3},
        {"predicates": {"x": 1}},
    ):
        with pytest.raises(Exception):  # noqa: B017 - several validation error types
            EvalConfig(**bad)  # type: ignore[arg-type]


def test_max_line_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("DISSEQT_REPORT_MAX_LINES", raising=False)
    assert resolve_max_lines(None) == 40
    assert resolve_max_lines(7) == 7 and resolve_max_lines(0) is None
    assert resolve_max_lines(None, 12) == 12 and resolve_max_lines(None, 0) is None
    monkeypatch.setenv("DISSEQT_REPORT_MAX_LINES", "oops")
    assert resolve_max_lines(None) == 40  # an unparsable value falls back
    monkeypatch.setenv("DISSEQT_REPORT_MAX_LINES", "9")
    assert resolve_max_lines(None) == 9 and resolve_max_lines(3) == 3  # explicit beats env


# ---------------------------------------------------------------- Ref sources


def solve(tools: list[ToolExpect], *steps: Any, fakes: tuple[Any, ...] = (), **kw: Any) -> Any:
    return assign(tuple(tools), run_of(*steps, **kw), EvalConfig(), fakes)


def test_input_refs_can_target_a_specific_turn() -> None:
    tools = [ToolExpect("t", {"x": Ref("input:1")})]
    run_kw = {"follow_ups": ("order A12 please",), "outputs": ("a", "b")}
    ok = solve(tools, step(1, "t", {"x": "a12"}, turn=1), **run_kw)
    assert ok.pairs[0].eval.score == 1.0
    bad = solve(tools, step(1, "t", {"x": "zzz"}, turn=1), **run_kw)
    assert bad.pairs[0].eval.leaves[0].code == "ungrounded_value"
    missing_turn = solve([ToolExpect("t", {"x": Ref("input:5")})], step(1, "t", {"x": "a"}))
    assert missing_turn.pairs[0].eval.score == 0.0


def test_input_refs_use_every_user_message_of_a_history_input() -> None:
    from disseqt_agent_testing.models import AgentRun, Message

    run = AgentRun(
        steps=(step(1, "t", {"x": "A12"}),),
        input=(Message("user", "my order is A12"), Message("assistant", "ok"), Message("user", "refund")),
        final_text="d", final_answer="d", turn_outputs=("d",),
    )  # fmt: skip
    a = assign((ToolExpect("t", {"x": Ref("input")}),), run, EvalConfig())
    assert a.pairs[0].eval.score == 1.0


def test_fake_refs_with_paths_and_failures() -> None:
    spec = FakeSpec("lookup", default=FakeResponse(returns={"items": [{"id": "A1"}]}))
    ok = solve(
        [ToolExpect("t", {"x": Ref("fake:lookup.items[0].id")})],
        step(1, "t", {"x": "A1"}),
        fakes=(spec,),
    )
    assert ok.pairs[0].eval.score == 1.0
    gone = solve(
        [ToolExpect("t", {"x": Ref("fake:lookup.nope")})], step(1, "t", {"x": "A1"}), fakes=(spec,)
    )
    assert gone.pairs[0].eval.leaves[0].details["reason"] == "path missing"
    unknown = solve(
        [ToolExpect("t", {"x": Ref("fake:ghost.v")})], step(1, "t", {"x": "A1"}), fakes=(spec,)
    )
    assert (
        unknown.pairs[0].eval.errored
        and "no fake named 'ghost'" in unknown.pairs[0].eval.leaves[0].reason
    )


def test_a_ref_to_a_tool_name_glob_and_an_args_path() -> None:
    steps = (step(1, "get_a", {"id": "Q1"}, effect="read"), step(2, "use", {"x": "Q1"}))
    ok = solve([ToolExpect("use", {"x": Ref("call:get_*.args.id")})], *steps)
    assert ok.pairs[0].eval.score == 1.0
    miss = solve([ToolExpect("use", {"x": Ref("call:zzz.args.id")})], *steps)
    assert miss.pairs[0].eval.leaves[0].details["reason"] == "source not matched"


# ---------------------------------------------------------------- rendering of unusual results


def test_budget_and_clarify_lines_for_skipped_and_errored_components() -> None:
    rep = evaluate(
        run_of(step(1, "a")), Expect(max_llm_calls=1, max_steps=1, max_tool_calls={"b": 1})
    )
    text = rep.render(max_lines=0)
    assert "budget" in text and "data_unavailable" in text and "max_tool_calls" in text


def test_default_lines_for_a_metric_with_no_renderer_and_unknown_codes() -> None:
    ok = MetricResult("custom", status="ok", value=1.0, passed=True)
    bad = MetricResult(
        "custom", status="ok", value=0.0, passed=False, reason_code="weird.code", reason="nope"
    )
    skipped = MetricResult("custom", status="skipped", reason_code="data_unavailable", reason="why")
    assert default_lines(ok)[0].startswith("custom")
    assert "weird.code" in default_lines(bad)[0] and "FAILED" in default_lines(bad)[0]
    assert "data_unavailable" in default_lines(skipped)[0]
    assert "custom" in render(
        {"checks": [ok.to_dict()], "outcome": "passed", "case_id": "c", "summary": {}}
    )


def test_every_metric_renderer_handles_every_status() -> None:
    from disseqt_agent_testing.assertions.render import LINES

    for name, fn in LINES.items():
        for status in ("ok-pass", "ok-fail", "skipped", "error"):
            if status == "ok-pass":
                r = MetricResult(name, status="ok", value=1.0, passed=True)
            elif status == "ok-fail":
                r = MetricResult(
                    name, status="ok", value=0.0, passed=False, reason_code="x.y", reason="r"
                )
            else:
                r = MetricResult(name, status=status, reason_code="data_unavailable", reason="why")  # type: ignore[arg-type]
            assert fn(r), (name, status)


def test_assert_agent_with_a_bad_threshold_raises_before_scoring() -> None:
    with pytest.raises(ExpectConfigError):
        assert_agent(run_of(), Expect(min_scores={"nope": 1.0}))


def test_aevaluate_propagates_config_errors() -> None:
    import asyncio

    with pytest.raises(ExpectConfigError):
        asyncio.run(aevaluate(run_of(), Expect(min_scores={"nope": 1.0})))
