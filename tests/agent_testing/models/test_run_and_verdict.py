"""Step, Usage, AgentRun, RunResult, StepEvent, ToolInfo, ErrorInfo, Violation, Verdict."""

import dataclasses

import pytest

from disseqt_agent_testing.models import (
    RUN_STATUSES,
    STEP_KIND_TO_SPAN_KIND,
    STEP_KINDS,
    VIOLATION_CODES,
    AgentRun,
    ErrorInfo,
    FrozenDict,
    Message,
    ModelValidationError,
    RunResult,
    Step,
    StepEvent,
    ToolInfo,
    Usage,
    Verdict,
    Violation,
)


def tool_step(seq=1, name="get_order", **kwargs):
    kwargs.setdefault("effect", "read")
    return Step(seq, "tool", name, **kwargs)


class TestStep:
    def test_defaults(self):
        step = tool_step()
        assert step.call_id == "call-1" and step.args == {} and step.result is None
        assert step.ts == 0.0 and step.duration_ms is None and step.turn == 0
        assert step.parallel_group is None and step.usage is None and step.lossy is False

    def test_call_id_follows_seq_unless_given(self):
        assert tool_step(7).call_id == "call-7"
        assert tool_step(7, call_id="x-1").call_id == "x-1"

    def test_values_are_frozen(self):
        step = tool_step(args={"a": [1]}, result={"b": 2})
        assert isinstance(step.args, FrozenDict) and step.args["a"] == (1,)
        assert isinstance(step.result, FrozenDict)
        hash(step)

    def test_layout(self):
        with pytest.raises(TypeError):
            Step(1, "tool", "x", "call-1")  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"seq": 0, "kind": "tool", "name": "x", "effect": "read"}, "Step.seq"),
            ({"seq": 1, "kind": "rag", "name": "x", "effect": "read"}, "Step.kind"),
            ({"seq": 1, "kind": "tool", "name": "", "effect": "read"}, "Step.name"),
            ({"seq": 1, "kind": "tool", "name": "x"}, "effect: is required"),
            ({"seq": 1, "kind": "subagent", "name": "x"}, "effect: is required"),
            (
                {"seq": 1, "kind": "model", "name": "m", "effect": "read"},
                "must be None for 'model'",
            ),
            ({"seq": 1, "kind": "tool", "name": "x", "effect": "maybe"}, "Step.effect"),
            ({"seq": 1, "kind": "tool", "name": "x", "effect": "read", "ts": -1}, "Step.ts"),
            ({"seq": 1, "kind": "tool", "name": "x", "effect": "read", "turn": -1}, "Step.turn"),
            (
                {"seq": 1, "kind": "tool", "name": "x", "effect": "read", "duration_ms": -1},
                "duration_ms",
            ),
            ({"seq": 1, "kind": "tool", "name": "x", "effect": "read", "usage": 3}, "Step.usage"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ModelValidationError, match=match):
            Step(**kwargs)

    def test_model_step_has_no_effect(self):
        step = Step(1, "model", "fake-llm", usage=Usage(input_tokens=10, output_tokens=5))
        assert step.effect is None
        assert step.usage == Usage(input_tokens=10, output_tokens=5)

    def test_kinds_map_to_span_kinds(self):
        from disseqt_agentic_sdk.enums.span_kind import SpanKind

        assert set(STEP_KIND_TO_SPAN_KIND) == set(STEP_KINDS)
        for span_kind in STEP_KIND_TO_SPAN_KIND.values():
            assert span_kind in SpanKind.__members__


class TestUsage:
    def test_defaults_and_validation(self):
        usage = Usage()
        assert (usage.input_tokens, usage.output_tokens) == (0, 0)
        assert (usage.latency_s, usage.cost_usd) == (0.0, 0.0)
        for kwargs in (
            {"input_tokens": -1},
            {"output_tokens": True},
            {"latency_s": float("nan")},
            {"cost_usd": -0.1},
        ):
            with pytest.raises(ModelValidationError, match="Usage"):
                Usage(**kwargs)

    def test_keyword_only(self):
        with pytest.raises(TypeError):
            Usage(1, 2)  # type: ignore[misc]


class TestSmallRecords:
    def test_error_info(self):
        info = ErrorInfo("ValueError", "builtins", "ValueError: boom", traceback="...")
        assert info.message == "ValueError: boom"
        with pytest.raises(ModelValidationError, match="ErrorInfo.message"):
            ErrorInfo("ValueError", "builtins", "")

    def test_violation_codes(self):
        assert Violation("unexpected_call", "get_order({}) matched no response", fake="get_order")
        with pytest.raises(ModelValidationError, match="Violation.code"):
            Violation("made_up", "x")
        assert "unexpected_call" in VIOLATION_CODES

    def test_tool_info(self):
        info = ToolInfo("get_order", "tool", "read", description="d", schema={"type": "object"})
        assert isinstance(info.schema, FrozenDict)
        with pytest.raises(ModelValidationError, match="ToolInfo.kind"):
            ToolInfo("x", "mcp", "read")  # type: ignore[arg-type]

    def test_step_event(self):
        event = StepEvent("start", "t1", 1, "call-1", "tool", "get_order", 0, args={"a": 1})
        assert event.args == {"a": 1}
        with pytest.raises(ModelValidationError, match="StepEvent.phase"):
            StepEvent("middle", "t1", 1, "call-1", "tool", "x", 0)


class TestAgentRun:
    def test_defaults(self):
        run = AgentRun()
        assert run.steps == () and run.final_answer is None and run.model_steps_captured is False
        assert run.state == {} and run.usage == Usage() and run.error is None

    def test_all_keyword_only(self):
        with pytest.raises(TypeError):
            AgentRun("case")  # type: ignore[misc]

    def test_helpers_exclude_model_steps(self):
        run = AgentRun(
            steps=(
                Step(1, "model", "fake-llm"),
                tool_step(2, "get_order"),
                Step(3, "subagent", "faq_agent", effect="read"),
            )
        )
        assert [s.seq for s in run.tool_steps()] == [2, 3]
        assert run.tool_names() == ("get_order", "faq_agent")

    def test_steps_must_be_sorted_and_unique(self):
        with pytest.raises(
            ModelValidationError, match=r"sorted by strictly increasing seq, got \[2, 1\]"
        ):
            AgentRun(steps=(tool_step(2), tool_step(1)))
        with pytest.raises(ModelValidationError, match="strictly increasing"):
            AgentRun(steps=(tool_step(1), tool_step(1)))

    def test_coercion_and_checks(self):
        run = AgentRun(
            input=[Message("user", "hi")],
            steps=[tool_step()],
            turn_outputs=["a", {"k": [1]}],
            final_answer={"k": 1},
            final_text='{"k": 1}',
        )
        assert isinstance(run.steps, tuple) and isinstance(run.input, tuple)
        assert run.turn_outputs[1] == {"k": (1,)}
        with pytest.raises(ModelValidationError, match=r"steps\[0\]"):
            AgentRun(steps=("x",))  # type: ignore[arg-type]
        with pytest.raises(ModelValidationError, match="AgentRun.final_answer"):
            AgentRun(final_answer=object())

    def test_frozen_and_hashable(self):
        run = AgentRun(steps=(tool_step(),))
        hash(run)
        with pytest.raises(dataclasses.FrozenInstanceError):
            run.steps = ()  # type: ignore[misc]


class TestRunResult:
    def error(self):
        return ErrorInfo("ValueError", "builtins", "ValueError: boom")

    def test_finished_has_no_error(self):
        result = RunResult("finished", AgentRun())
        assert result.error is None and result.violations == () and result.warnings == ()

    @pytest.mark.parametrize("status", ["crashed", "setup_failed"])
    def test_failures_need_an_error(self, status):
        with pytest.raises(ModelValidationError, match="RunResult.error: is required"):
            RunResult(status, AgentRun())
        assert RunResult(status, AgentRun(), error=self.error()).error is not None

    @pytest.mark.parametrize("status", ["finished", "timed_out", "aborted"])
    def test_other_statuses_forbid_an_error(self, status):
        with pytest.raises(ModelValidationError, match="RunResult.error: must be None"):
            RunResult(status, AgentRun(), error=self.error())

    def test_validation(self):
        with pytest.raises(ModelValidationError, match="RunResult.status"):
            RunResult("done", AgentRun())
        with pytest.raises(ModelValidationError, match="RunResult.run"):
            RunResult("finished", "x")  # type: ignore[arg-type]
        with pytest.raises(ModelValidationError, match=r"violations\[0\]"):
            RunResult("finished", AgentRun(), violations=("x",))  # type: ignore[arg-type]
        assert set(RUN_STATUSES) == {"finished", "crashed", "setup_failed", "timed_out", "aborted"}


class TestVerdict:
    def test_defaults(self):
        verdict = Verdict("tool_order", False, 0.0, "expected a before b")
        assert verdict.step_index is None and verdict.scorer == "deterministic"
        assert verdict.extra == {}

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            (("", True, 1.0, "r"), "Verdict.dimension"),
            (("d", "yes", 1.0, "r"), "Verdict.passed"),
            (("d", True, 1.5, "r"), "Verdict.score"),
            (("d", True, -0.1, "r"), "Verdict.score"),
            (("d", True, float("nan"), "r"), "Verdict.score"),
            (("d", True, 1.0, ""), "Verdict.reason"),
        ],
    )
    def test_validation(self, args, match):
        with pytest.raises(ModelValidationError, match=match):
            Verdict(*args)

    def test_other_fields(self):
        with pytest.raises(ModelValidationError, match="Verdict.step_index"):
            Verdict("d", True, 1.0, "r", step_index=-1)
        with pytest.raises(ModelValidationError, match="Verdict.scorer"):
            Verdict("d", True, 1.0, "r", scorer="llm")  # type: ignore[arg-type]
        assert Verdict("d", True, 1, "r").score == 1.0
