"""ToolExpect, ForbiddenCall, Expect, FakeResponse, FakeSpec, AgentTestCase."""

import dataclasses

import pytest

from disseqt_agent_testing.models import (
    AgentTestCase,
    Decision,
    Eq,
    Expect,
    FakeResponse,
    FakeSpec,
    ForbiddenCall,
    FrozenDict,
    Message,
    ModelValidationError,
    Ref,
    Regex,
    ToolExpect,
)


def refund_fakes():
    return (
        FakeSpec("get_order", effect="read"),
        FakeSpec("check_policy", effect="read"),
        FakeSpec("issue_refund"),
    )


class TestToolExpect:
    def test_layout_name_and_args_positional_rest_keyword_only(self):
        tool = ToolExpect("issue_refund", {"order_id": "A12"}, after=("lookup",))
        assert tool.args == {"order_id": Eq("A12")} and tool.after == ("lookup",)
        with pytest.raises(TypeError):
            ToolExpect("x", {}, "subset")  # type: ignore[misc]

    def test_plain_values_and_matchers_are_normalized(self):
        tool = ToolExpect("x", {"a": 1, "b": Regex("^z"), "c": Ref("input")})
        assert tool.args == {"a": Eq(1), "b": Regex("^z"), "c": Ref("input")}
        assert isinstance(tool.args, FrozenDict)

    def test_defaults(self):
        tool = ToolExpect("x")
        assert (tool.args_mode, tool.count, tool.id, tool.after, tool.turn) == (
            "subset",
            None,
            None,
            (),
            None,
        )
        assert tool.defaults == {} and tool.extra == {}

    def test_two_instances_never_share_default_containers(self):
        a, b = ToolExpect("x"), ToolExpect("y")
        assert a.args is not b.args and a.extra is not b.extra

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(2, (2, 2)), (1, (1, 1)), ((1, None), (1, None)), ([0, 3], (0, 3)), (None, None)],
    )
    def test_count_normalization(self, count, expected):
        assert ToolExpect("x", count=count).count == expected

    def test_count_zero_points_to_forbidden(self):
        with pytest.raises(ModelValidationError, match="use Expect.forbidden"):
            ToolExpect("x", count=0)

    @pytest.mark.parametrize("count", [-1, True, (2, 1), (-1, 3), (1, 2, 3), "2", (1, 0)])
    def test_bad_counts(self, count):
        with pytest.raises(ModelValidationError, match="ToolExpect.count"):
            ToolExpect("x", count=count)

    @pytest.mark.parametrize("bad_id", ["", "a b", "a.b", "x" * 65, 3])
    def test_ids_must_be_ref_friendly(self, bad_id):
        with pytest.raises(ModelValidationError, match="ToolExpect.id"):
            ToolExpect("x", id=bad_id)

    def test_other_field_checks(self):
        with pytest.raises(ModelValidationError, match="ToolExpect.args_mode"):
            ToolExpect("x", args_mode="loose")  # type: ignore[arg-type]
        with pytest.raises(ModelValidationError, match="ToolExpect.turn"):
            ToolExpect("x", turn=-1)
        with pytest.raises(ModelValidationError, match="ToolExpect.name"):
            ToolExpect("")

    def test_immutable(self):
        tool = ToolExpect("x", {"a": 1})
        with pytest.raises(dataclasses.FrozenInstanceError):
            tool.name = "y"  # type: ignore[misc]
        with pytest.raises(TypeError):
            tool.args["b"] = Eq(2)


class TestForbiddenCall:
    def test_defaults_and_positional(self):
        veto = ForbiddenCall("issue_refund")
        assert veto.args is None and veto.unless_after is None and veto.reason == ""
        assert ForbiddenCall("x", {"a": 1}).args == {"a": Eq(1)}

    def test_scoring_time_matchers_are_rejected(self):
        with pytest.raises(ModelValidationError, match="matcher Ref is not allowed in"):
            ForbiddenCall("x", {"a": Ref("input")})


class TestFakeResponseAndSpec:
    def test_response_defaults_and_when(self):
        response = FakeResponse(when={"order_id": "A12"}, returns={"total": 59.99})
        assert response.when == {"order_id": Eq("A12")} and response.returns == {"total": 59.99}
        assert response.error is None and response.times is None and response.latency_s == 0.0

    def test_response_is_all_keyword_only(self):
        with pytest.raises(TypeError):
            FakeResponse({})  # type: ignore[misc]

    def test_error_and_returns_are_exclusive(self):
        with pytest.raises(ModelValidationError, match="mutually exclusive"):
            FakeResponse(error="down", returns=1)
        assert FakeResponse(error="down").error == "down"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"times": 0}, "FakeResponse.times"),
            ({"latency_s": -1}, "FakeResponse.latency_s"),
            ({"latency_s": float("inf")}, "FakeResponse.latency_s"),
            ({"returns": object()}, "FakeResponse.returns"),
            ({"when": {"a": Ref("input")}}, "matcher Ref is not allowed in"),
        ],
    )
    def test_response_validation(self, kwargs, match):
        with pytest.raises(ModelValidationError, match=match):
            FakeResponse(**kwargs)

    def test_spec_defaults_effect_is_write(self):
        spec = FakeSpec("issue_refund")
        assert spec.kind == "tool" and spec.effect == "write" and spec.schema is None
        assert spec.responses == () and spec.default is None

    def test_spec_checks(self):
        with pytest.raises(ModelValidationError, match="FakeSpec.kind"):
            FakeSpec("x", kind="mcp")  # type: ignore[arg-type]
        with pytest.raises(ModelValidationError, match="FakeSpec.effect"):
            FakeSpec("x", effect="maybe")  # type: ignore[arg-type]
        with pytest.raises(ModelValidationError, match="type must be 'object'"):
            FakeSpec("x", schema={"type": "array"})
        with pytest.raises(ModelValidationError, match=r"responses\[0\]"):
            FakeSpec("x", responses=("nope",))  # type: ignore[arg-type]
        with pytest.raises(ModelValidationError, match="FakeSpec.default"):
            FakeSpec("x", default={"returns": 1})  # type: ignore[arg-type]

    def test_schema_is_frozen_and_has_no_handler(self):
        spec = FakeSpec("x", schema={"type": "object", "properties": {"a": {}}, "required": ["a"]})
        assert isinstance(spec.schema, FrozenDict) and spec.schema["required"] == ("a",)
        assert "handler" not in {f.name for f in dataclasses.fields(FakeSpec)}


class TestExpect:
    def test_defaults(self):
        expect = Expect()
        assert expect.decisions == () and expect.tools == () and expect.order == "any"
        assert expect.extra_calls == "allow_reads" and expect.allowed_tools is None
        assert expect.max_steps is None and expect.min_scores == {}

    def test_single_decision_is_wrapped_and_strings_are_accepted(self):
        assert Expect(decision=Decision.ASK).decision == (Decision.ASK,)
        assert Expect(decision="refuse").decision == (Decision.REFUSE,)
        assert Expect(decision=["ask", Decision.REFUSE]).decision == (Decision.ASK, Decision.REFUSE)

    def test_bad_decision(self):
        with pytest.raises(ModelValidationError, match="Expect.decision: must be one of"):
            Expect(decision="maybe")
        with pytest.raises(ModelValidationError, match=r"Expect.decision\[1\]"):
            Expect(decision=["ask", "maybe"])

    def test_decision_by_turn_normalization(self):
        expect = Expect(decision="ask", decision_by_turn={2: "call", "1": ["ask", "refuse"]})
        assert list(expect.decision_by_turn) == ["1", "2"]
        assert expect.decision_by_turn["1"] == (Decision.ASK, Decision.REFUSE)
        assert expect.turn_decisions == {
            0: (Decision.ASK,),
            1: (Decision.ASK, Decision.REFUSE),
            2: (Decision.CALL,),
        }

    @pytest.mark.parametrize("by_turn", [{0: "ask"}, {"0": "ask"}])
    def test_turn_zero_is_set_by_decision(self, by_turn):
        with pytest.raises(ModelValidationError, match="turn 0 is set by 'decision'"):
            Expect(decision_by_turn=by_turn)

    @pytest.mark.parametrize("by_turn", [{"x": "ask"}, {True: "ask"}, {1: []}, [1]])
    def test_bad_decision_by_turn(self, by_turn):
        with pytest.raises(ModelValidationError, match="Expect.decision_by_turn"):
            Expect(decision_by_turn=by_turn)

    def test_call_needs_tools_or_allowed_tools(self):
        with pytest.raises(ModelValidationError, match="'call' needs tools or allowed_tools"):
            Expect(decision="call")
        Expect(decision="call", tools=(ToolExpect("x"),))
        Expect(decision="call", allowed_tools=("x",))

    def test_by_turn_call_does_not_need_tools(self):
        Expect(decision="ask", decision_by_turn={1: "call"}, forbidden=(ForbiddenCall("x"),))

    def test_ids_must_be_unique(self):
        with pytest.raises(ModelValidationError, match=r"repeated: \['a'\]"):
            Expect(tools=(ToolExpect("x", id="a"), ToolExpect("y", id="a")))

    def test_after_must_name_an_existing_id(self):
        with pytest.raises(
            ModelValidationError, match="after 'nope' does not match any ToolExpect.id"
        ):
            Expect(tools=(ToolExpect("x", id="a", after=("nope",)),), order="partial")

    def test_a_tool_with_after_does_not_need_its_own_id(self):
        # Nothing can point at an id-less tool, so it cannot be part of a cycle.
        expect = Expect(
            tools=(ToolExpect("x", id="a"), ToolExpect("y", {"k": 1}, after=("a",))),
            order="partial",
        )
        assert expect.tools[1].id is None and expect.tools[1].after == ("a",)

    def test_after_requires_partial_order(self):
        with pytest.raises(ModelValidationError, match="must be 'partial'"):
            Expect(tools=(ToolExpect("x", id="a"), ToolExpect("y", id="b", after=("a",))))

    def test_cycle_is_reported_with_its_path(self):
        tools = (
            ToolExpect("x", id="a", after=("b",)),
            ToolExpect("y", id="b", after=("a",)),
        )
        with pytest.raises(
            ModelValidationError, match=r"cycle in after: (a -> b -> a|b -> a -> b)"
        ):
            Expect(tools=tools, order="partial")

    def test_valid_dag(self):
        Expect(
            tools=(
                ToolExpect("a", id="a"),
                ToolExpect("b", id="b", after=("a",)),
                ToolExpect("c", id="c", after=("a",)),
                ToolExpect("d", id="d", after=("b", "c")),
            ),
            order="partial",
        )

    def test_forbidden_unless_after_must_exist(self):
        with pytest.raises(ModelValidationError, match="unless_after"):
            Expect(forbidden=(ForbiddenCall("x", unless_after="verify"),))
        Expect(
            tools=(ToolExpect("v", id="verify"),),
            forbidden=(ForbiddenCall("x", unless_after="verify"),),
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_steps": 0}, "Expect.max_steps"),
            ({"max_tokens": 0}, "Expect.max_tokens"),
            ({"max_latency_s": 0}, "Expect.max_latency_s"),
            ({"max_tool_calls": 0}, "Expect.max_tool_calls"),
            ({"max_tool_calls": {"x": 0}}, "Expect.max_tool_calls"),
            ({"min_scores": {"tool_args": 1.5}}, "Expect.min_scores"),
            ({"order": "sorted"}, "Expect.order"),
            ({"extra_calls": "sometimes"}, "Expect.extra_calls"),
            ({"answer_contains": (1,)}, "Expect.answer_contains"),
            ({"tools": ("x",)}, r"Expect.tools\[0\]"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ModelValidationError, match=match):
            Expect(**kwargs)

    def test_budgets_and_text(self):
        expect = Expect(
            max_tool_calls={"search_*": 2},
            max_latency_s=5,
            answer_contains=("refunded", Regex("59\\.99")),
            answer_excludes=("sorry",),
            min_scores={"tool_args": 0.5},
        )
        assert expect.max_tool_calls == {"search_*": 2} and expect.max_latency_s == 5.0
        assert expect.answer_contains == ("refunded", Regex("59\\.99"))

    def test_hashable(self):
        hash(Expect(decision="ask", tools=(ToolExpect("x", {"a": 1}),)))


class TestAgentTestCase:
    def case(self, **kwargs):
        return AgentTestCase("refund_happy", "Refund order A12", **kwargs)

    def test_string_input_becomes_one_user_message(self):
        case = self.case()
        assert case.input == (Message("user", "Refund order A12"),) == case.messages
        assert case.repeats == 1 and case.pass_k is None and case.pass_k_threshold == 1.0

    def test_message_input(self):
        history = (
            Message("user", "hi"),
            Message("assistant", "hello"),
            Message("user", "refund"),
        )
        assert AgentTestCase("c", history).messages == history

    @pytest.mark.parametrize("bad", ["", "-x", "a b", "x" * 129, "a/b"])
    def test_id_pattern(self, bad):
        with pytest.raises(ModelValidationError, match="AgentTestCase.id"):
            AgentTestCase(bad, "hi")

    def test_input_rules(self):
        with pytest.raises(ModelValidationError, match="last input message must be from the user"):
            AgentTestCase("c", (Message("user", "hi"), Message("assistant", "yo")))
        with pytest.raises(ModelValidationError, match="must not be empty"):
            AgentTestCase("c", ())
        with pytest.raises(ModelValidationError, match=r"input\[0\]"):
            AgentTestCase("c", ("hi",))  # type: ignore[arg-type]

    def test_repeats_and_pass_k(self):
        with pytest.raises(ModelValidationError, match=r"pass_k \(5\) cannot exceed repeats \(3\)"):
            self.case(repeats=3, pass_k=5)
        with pytest.raises(ModelValidationError, match="repeats"):
            self.case(repeats=0)
        with pytest.raises(ModelValidationError, match="pass_k_threshold"):
            self.case(pass_k_threshold=0)
        with pytest.raises(ModelValidationError, match="pass_k_threshold"):
            self.case(pass_k_threshold=1.01)
        assert self.case(repeats=5, pass_k=3, pass_k_threshold=0.8).pass_k_threshold == 0.8

    def test_tags(self):
        assert self.case(tags=["smoke", "refund"]).tags == ("smoke", "refund")
        with pytest.raises(ModelValidationError, match="reserved"):
            self.case(tags=("skip",))
        with pytest.raises(ModelValidationError, match="must match"):
            self.case(tags=("my-tag",))

    def test_follow_ups(self):
        assert self.case(follow_ups=["It's A12"]).follow_ups == ("It's A12",)
        with pytest.raises(ModelValidationError, match="follow_ups"):
            self.case(follow_ups=("",))

    def test_duplicate_fakes(self):
        with pytest.raises(ModelValidationError, match="duplicate fake name 'x'"):
            self.case(fakes=(FakeSpec("x"), FakeSpec("x")))

    def test_expected_tools_must_have_fakes_when_fakes_are_given(self):
        expect = Expect(tools=(ToolExpect("get_order"), ToolExpect("lookup")))
        with pytest.raises(
            ModelValidationError,
            match=r"expected tool 'lookup' has no fake; known: \['check_policy', 'get_order', 'issue_refund'\]",
        ):
            self.case(fakes=refund_fakes(), expect=expect)
        with pytest.raises(ModelValidationError, match="Expect.forbidden"):
            self.case(fakes=refund_fakes(), expect=Expect(forbidden=(ForbiddenCall("delete"),)))

    def test_globs_must_match_a_fake(self):
        self.case(fakes=refund_fakes(), expect=Expect(tools=(ToolExpect("get_*"),)))
        with pytest.raises(ModelValidationError, match="glob 'zzz_\\*' matches no fake"):
            self.case(fakes=refund_fakes(), expect=Expect(tools=(ToolExpect("zzz_*"),)))

    def test_tool_names_are_not_checked_without_fakes(self):
        self.case(expect=Expect(tools=(ToolExpect("anything"),)))

    def test_ask_cannot_expect_a_write_tool(self):
        expect = Expect(decision="ask", tools=(ToolExpect("issue_refund"),))
        with pytest.raises(
            ModelValidationError,
            match="decision 'ask' on turn 0 cannot expect write tool 'issue_refund'",
        ):
            self.case(fakes=refund_fakes(), expect=expect)

    def test_read_tools_are_fine_before_asking(self):
        self.case(
            fakes=refund_fakes(), expect=Expect(decision="ask", tools=(ToolExpect("get_order"),))
        )

    def test_write_tool_on_a_later_turn_is_fine(self):
        expect = Expect(
            decision="ask",
            decision_by_turn={1: "call"},
            tools=(ToolExpect("issue_refund", turn=1),),
        )
        self.case(fakes=refund_fakes(), expect=expect, follow_ups=("A12",))

    def test_write_check_is_skipped_when_the_fake_is_unknown(self):
        self.case(expect=Expect(decision="ask", tools=(ToolExpect("issue_refund"),)))

    def test_immutable_and_hashable(self):
        case = self.case(fakes=refund_fakes())
        hash(case)
        with pytest.raises(dataclasses.FrozenInstanceError):
            case.repeats = 3  # type: ignore[misc]

    def test_extra_bag(self):
        case = self.case(extra={"acme.ticket": "T-1"})
        assert case.extra == {"acme.ticket": "T-1"}
        with pytest.raises(ModelValidationError, match="AgentTestCase.extra"):
            self.case(extra={"x": float("nan")})


from disseqt_agent_testing.models import (  # noqa: E402
    AnswerQualityExpect,
    ClarifyExpect,
    GroundednessExpect,
    LoopExpect,
    PlanExpect,
    RefuseExpect,
    RouteExpect,
    ScopeExpect,
    Semantic,
)


class TestToolExpect1B:
    def test_defaults(self):
        tool = ToolExpect("x")
        assert tool.ignore == () and tool.same_step_as is None and tool.weight == 1.0

    def test_values(self):
        tool = ToolExpect("x", ignore=["trace_*"], same_step_as="other", weight=2)
        assert tool.ignore == ("trace_*",) and tool.same_step_as == "other" and tool.weight == 2.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"weight": 0}, "ToolExpect.weight"),
            ({"weight": -1}, "ToolExpect.weight"),
            ({"weight": float("nan")}, "ToolExpect.weight"),
            ({"same_step_as": "bad id"}, "ToolExpect.same_step_as"),
            ({"ignore": [""]}, "ToolExpect.ignore"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ModelValidationError, match=match):
            ToolExpect("x", **kwargs)

    def test_same_step_as_must_name_another_tool(self):
        with pytest.raises(ModelValidationError, match=r"same_step_as: 'zzz' does not match"):
            Expect(tools=(ToolExpect("x", id="a", same_step_as="zzz"),))
        with pytest.raises(ModelValidationError, match="cannot refer to itself"):
            Expect(tools=(ToolExpect("x", id="a", same_step_as="a"),))
        Expect(tools=(ToolExpect("x", id="a"), ToolExpect("y", same_step_as="a")))


class TestSubBlocks:
    def test_route_expect(self):
        assert RouteExpect(path=["a", "b"]).mode == "in_order"
        assert RouteExpect(mode="exact").path == () and RouteExpect(mode="none").path == ()
        with pytest.raises(ModelValidationError, match="must be empty when mode is 'none'"):
            RouteExpect(path=("a",), mode="none")
        for mode in ("in_order", "first", "any"):
            with pytest.raises(ModelValidationError, match="must not be empty"):
                RouteExpect(mode=mode)
        with pytest.raises(ModelValidationError, match="RouteExpect.mode"):
            RouteExpect(path=("a",), mode="sometimes")

    def test_loop_expect_defaults_and_bounds(self):
        loops = LoopExpect()
        assert (loops.max_identical_calls, loops.max_consecutive_identical) == (3, 2)
        assert (loops.detect_cycles, loops.max_period, loops.min_repeats) == (True, 4, 2)
        assert loops.key == "name_args" and loops.max_retries_after_error == 2
        assert LoopExpect(max_identical_calls=None).max_identical_calls is None
        for kwargs in (
            {"max_identical_calls": 0}, {"max_consecutive_identical": 0}, {"max_period": 1},
            {"min_repeats": 1}, {"key": "args"}, {"max_retries_after_error": -1},
            {"detect_cycles": "yes"},
        ):  # fmt: skip
            with pytest.raises(ModelValidationError, match="LoopExpect"):
                LoopExpect(**kwargs)

    def test_plan_expect_constraints_are_predicate_names(self):
        assert PlanExpect(constraints=["refund_after_lookup"]).constraints == (
            "refund_after_lookup",
        )
        with pytest.raises(ModelValidationError, match="registered predicate name"):
            PlanExpect(constraints=["not a name"])
        with pytest.raises(ModelValidationError, match="optimal_tool_calls"):
            PlanExpect(optimal_tool_calls=0)

    def test_clarify_refuse_scope(self):
        clarify = ClarifyExpect(must_mention_any=["order id"], max_questions=1)
        assert clarify.allow_read_calls_before is True and clarify.judge_relevance is False
        with pytest.raises(ModelValidationError, match="max_questions"):
            ClarifyExpect(max_questions=0)
        assert RefuseExpect().offer_alternative is None
        assert RefuseExpect(offer_alternative=False).offer_alternative is False
        with pytest.raises(ModelValidationError, match="offer_alternative"):
            RefuseExpect(offer_alternative="yes")
        assert ScopeExpect(True).topics == ()
        with pytest.raises(ModelValidationError, match="in_scope"):
            ScopeExpect("yes")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            ScopeExpect(True, ("t",))  # type: ignore[misc]

    def test_answer_quality_and_groundedness(self):
        quality = AnswerQualityExpect("Refund issued")
        assert quality.accept == "correct" and quality.criteria == ()
        with pytest.raises(ModelValidationError, match="reference"):
            AnswerQualityExpect("")
        with pytest.raises(ModelValidationError, match="accept"):
            AnswerQualityExpect("x", accept="incorrect")
        grounded = GroundednessExpect()
        assert grounded.sources == ("*",) and grounded.include_user_statements is True
        assert grounded.allow_unsupported is False
        with pytest.raises(ModelValidationError, match="sources"):
            GroundednessExpect(sources=())

    def test_blocks_are_immutable_hashable_with_extra(self):
        for block in (RouteExpect(mode="exact"), LoopExpect(), PlanExpect(), ClarifyExpect(),
                      RefuseExpect(), ScopeExpect(False), AnswerQualityExpect("x"),
                      GroundednessExpect()):  # fmt: skip
            hash(block)
            assert block.extra == {} and isinstance(block.extra, FrozenDict)
            with pytest.raises(dataclasses.FrozenInstanceError):
                block.extra = {}  # type: ignore[misc]


class TestExpect1B:
    def test_defaults(self):
        expect = Expect()
        assert expect.max_llm_calls is None and expect.routes is None and expect.loops is None
        assert expect.intents == () and expect.intent_labels == () and expect.intent_source is None
        assert expect.clarify is None and expect.refuse is None and expect.scope is None
        assert expect.plan is None and expect.answer_quality is None and expect.grounded is None

    def test_routes_shorthand(self):
        assert Expect(routes=("billing", "refunds")).routes == RouteExpect(
            path=("billing", "refunds")
        )
        with pytest.raises(ModelValidationError, match="Expect.routes"):
            Expect(routes="billing")  # type: ignore[arg-type]

    def test_empty_route_tuple_is_rejected_because_in_order_needs_a_path(self):
        with pytest.raises(ModelValidationError, match="RouteExpect.path: must not be empty"):
            Expect(routes=())
        assert Expect(routes=RouteExpect(mode="none")).routes.mode == "none"  # type: ignore[union-attr]

    def test_grounded_shorthand(self):
        assert Expect(grounded=True).grounded == GroundednessExpect()
        assert Expect(grounded=False).grounded is None
        assert Expect(grounded=GroundednessExpect(allow_unsupported=True)).grounded.allow_unsupported  # type: ignore[union-attr]
        with pytest.raises(ModelValidationError, match="Expect.grounded"):
            Expect(grounded="yes")  # type: ignore[arg-type]

    def test_block_types_are_checked(self):
        for name in ("loops", "clarify", "refuse", "scope", "plan", "answer_quality"):
            with pytest.raises(ModelValidationError, match=f"Expect.{name}"):
                Expect(**{name: "nope"})

    def test_intent_rules(self):
        assert Expect(intent="billing", intent_labels=("billing", "other")).intents == ("billing",)
        assert Expect(intent=["a", "b"], intent_source="route").intents == ("a", "b")
        with pytest.raises(
            ModelValidationError, match="Expect.intent: needs intent_labels or intent_source"
        ):
            Expect(intent="billing")
        with pytest.raises(ModelValidationError, match="'returns' is not in intent_labels"):
            Expect(intent="returns", intent_labels=("billing",))

    @pytest.mark.parametrize(
        "source",
        ["route", "call:classify.args.label", "call:classify.result.intent.name", "answer.intent",
         "answer.items[0].label", "call:cls_*.result.label"],
    )  # fmt: skip
    def test_valid_intent_sources(self, source):
        assert Expect(intent_source=source).intent_source == source

    @pytest.mark.parametrize(
        "source",
        ["", "routes", "call:classify", "call:classify.args", "answer", "call:x.out.y", "input"],
    )
    def test_invalid_intent_sources(self, source):
        with pytest.raises(ModelValidationError, match="Expect.intent_source"):
            Expect(intent_source=source)

    def test_max_llm_calls(self):
        assert Expect(max_llm_calls=3).max_llm_calls == 3
        with pytest.raises(ModelValidationError, match="Expect.max_llm_calls"):
            Expect(max_llm_calls=0)


class TestJudgeConsistency:
    def case(self, expect, **kwargs):
        return AgentTestCase("c", "hi", expect=expect, **kwargs)

    @pytest.mark.parametrize(
        ("expect", "reason"),
        [
            (Expect(tools=(ToolExpect("x", {"a": Semantic("n")}),)), "a Semantic matcher"),
            (Expect(answer_contains=(Semantic("n"),)), "a Semantic matcher"),
            (Expect(clarify=ClarifyExpect(judge_relevance=True)), "clarify.judge_relevance"),
            (Expect(refuse=RefuseExpect(offer_alternative=False)), "refuse.offer_alternative"),
            (Expect(plan=PlanExpect(judge_quality=True)), "plan.judge_quality"),
            (Expect(answer_quality=AnswerQualityExpect("r")), "answer_quality"),
            (Expect(grounded=True), "grounded"),
            (Expect(intent="a", intent_labels=("a",)), "intent without intent_source"),
        ],
    )
    def test_needs_use_judge(self, expect, reason):
        with pytest.raises(
            ModelValidationError, match=f"AgentTestCase.use_judge: must be set because.*{reason}"
        ):
            self.case(expect)
        self.case(expect, use_judge=("semantic_arg",))

    def test_deterministic_cases_do_not_need_a_judge(self):
        self.case(
            Expect(
                intent="a",
                intent_source="route",
                clarify=ClarifyExpect(must_mention_any=("x",)),
                refuse=RefuseExpect(must_mention_any=("no",)),
                plan=PlanExpect(constraints=("c",)),
                loops=LoopExpect(),
                scope=ScopeExpect(True),
                routes=("a",),
            )
        )

    def test_semantic_inside_nested_matchers_is_found(self):
        from disseqt_agent_testing.models import DictMatch

        expect = Expect(tools=(ToolExpect("x", {"a": DictMatch({"b": Semantic("n")})}),))
        with pytest.raises(ModelValidationError, match="a Semantic matcher"):
            self.case(expect)

    def test_semantic_is_still_rejected_in_vetoes_and_fakes(self):
        with pytest.raises(ModelValidationError, match="matcher Semantic is not allowed in"):
            ForbiddenCall("x", {"a": Semantic("n")})
        with pytest.raises(ModelValidationError, match="matcher Semantic is not allowed in"):
            FakeResponse(when={"a": Semantic("n")})


class TestHostileInput:
    @pytest.mark.parametrize("value", ["a\n", "a\n\n", "\n"])
    def test_ids_tags_and_names_reject_a_trailing_newline(self, value):
        with pytest.raises(ModelValidationError, match="AgentTestCase.id"):
            AgentTestCase(value, "hi")
        with pytest.raises(ModelValidationError, match="AgentTestCase.tags"):
            AgentTestCase("c", "hi", tags=(value,))
        with pytest.raises(ModelValidationError, match="ToolExpect.id"):
            ToolExpect("x", id=value)
        with pytest.raises(ModelValidationError, match="ToolExpect.same_step_as"):
            ToolExpect("x", same_step_as=value)
        with pytest.raises(ModelValidationError, match="registered predicate name"):
            PlanExpect(constraints=(f"ok{value}",))

    def test_intent_source_rejects_a_trailing_newline(self):
        with pytest.raises(ModelValidationError, match="Expect.intent_source"):
            Expect(intent_source="route\n")
        with pytest.raises(ModelValidationError, match="Expect.intent_source"):
            Expect(intent_source="answer.label\n")


class TestNames:
    @pytest.mark.parametrize("name", ["a b", "a\tb", "a\nb", " ", "tool\x00", "x​y"[0:0] or "a b"])
    def test_authored_tool_and_fake_names_reject_whitespace_and_control_characters(self, name):
        for build in (ToolExpect, ForbiddenCall, FakeSpec):
            with pytest.raises(
                ModelValidationError, match="must not contain whitespace|must not be empty"
            ):
                build(name)

    @pytest.mark.parametrize("name", ["get_order", "search_*", "mcp.fs.read", "tool-1", "outils_é"])
    def test_ordinary_names_and_globs_are_accepted(self, name):
        for build in (ToolExpect, ForbiddenCall, FakeSpec):
            assert build(name).name == name

    def test_model_emitted_tool_names_stay_unrestricted(self):
        # A test must be able to model a hallucinated or invalid tool name coming from a model.
        from disseqt_agent_testing.models import MsgToolCall

        assert MsgToolCall("c1", "get order!").name == "get order!"
