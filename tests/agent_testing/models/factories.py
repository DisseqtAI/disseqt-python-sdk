"""Rich model instances shared by the round-trip, golden and extensibility tests."""

from disseqt_agent_testing.models import (
    Absent,
    AgentRun,
    AgentTestCase,
    AnswerQualityExpect,
    AnyValue,
    Approx,
    ClarifyExpect,
    Contains,
    DateEq,
    Decision,
    DictMatch,
    Eq,
    ErrorInfo,
    Expect,
    FakeResponse,
    FakeSpec,
    ForbiddenCall,
    GroundednessExpect,
    LoopExpect,
    Message,
    MsgToolCall,
    Normalized,
    OneOf,
    Optional_,
    PlanExpect,
    Predicate,
    Ref,
    RefuseExpect,
    Regex,
    RouteExpect,
    RunResult,
    ScopeExpect,
    Semantic,
    Step,
    StepEvent,
    ToolExpect,
    ToolInfo,
    TypeIs,
    Unordered,
    Usage,
    Verdict,
    Violation,
)


def rich_case() -> AgentTestCase:
    return AgentTestCase(
        "refund_rich",
        (
            Message("system", "You are a refund agent.", name="policy"),
            Message("user", "Hi"),
            Message(
                "assistant",
                tool_calls=(
                    MsgToolCall("c1", "get_order", arguments={"order_id": "A12", "n": [1, 2.5]}),
                    MsgToolCall("c2", "bad", raw_arguments="{", parse_error="Expecting value"),
                ),
            ),
            Message("tool", '{"total": 59.99}', tool_call_id="c1", name="get_order"),
            Message("tool", "", tool_call_id="c2"),
            Message("user", "Refund order A12, it arrived broken"),
        ),
        follow_ups=("It's A12", "thanks"),
        fakes=(
            FakeSpec(
                "get_order",
                kind="tool",
                description="Look up an order",
                effect="read",
                schema={
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
                responses=(
                    FakeResponse(
                        when={"order_id": "A12"}, returns={"order_id": "A12", "total": 59.99}
                    ),
                    FakeResponse(
                        when={"order_id": Regex("^B")}, error="order service down", times=1
                    ),
                    FakeResponse(returns=None, latency_s=0.25, extra={"acme.note": "slow"}),
                ),
                extra={"acme.owner": "refunds"},
            ),
            FakeSpec(
                "check_policy", effect="read", default=FakeResponse(returns={"eligible": True})
            ),
            FakeSpec(
                "issue_refund", default=FakeResponse(returns={"amount": 59.99, "status": "ok"})
            ),
            FakeSpec(
                "faq_agent", kind="subagent", effect="read", default=FakeResponse(returns="see FAQ")
            ),
        ),
        expect=Expect(
            decision=(Decision.CALL,),
            decision_by_turn={1: (Decision.ASK, Decision.REFUSE), 2: Decision.CALL},
            tools=(
                ToolExpect("get_order", {"order_id": "A12"}, id="lookup", count=(1, None), turn=0),
                ToolExpect(
                    "issue_refund",
                    {
                        "order_id": Ref("call:lookup.result.order_id"),
                        "amount": Approx(59.99, abs_tol=0.01),
                        "reason": OneOf(["damaged", "broken"]),
                        "note": Optional_(Regex("^x", flags=2)),
                        "dry_run": Absent(),
                        "trace": AnyValue(),
                        "n": Eq(1, coerce_numeric=False),
                        "meta": Eq({"$m": "looks like a matcher"}),
                    },
                    id="refund",
                    after=("lookup",),
                    args_mode="exact",
                    count=1,
                    turn=2,
                    defaults={"currency": "USD"},
                    extra={"acme.weight": 2},
                ),
            ),
            order="partial",
            extra_calls="forbid",
            allowed_tools=("get_*", "issue_refund"),
            forbidden=(
                ForbiddenCall("check_*", reason="policy lookups are internal"),
                ForbiddenCall("issue_refund", {"amount": 0}, unless_after="lookup"),
            ),
            max_steps=10,
            max_tool_calls={"get_*": 2, "issue_refund": 1},
            max_tokens=2000,
            max_latency_s=5.5,
            answer_contains=("refunded", Regex("59\\.99")),
            answer_excludes=("sorry",),
            min_scores={"tool_args": 0.5},
            extra={"acme.suite": "smoke"},
        ),
        tags=("smoke", "refund"),
        repeats=5,
        pass_k=3,
        pass_k_threshold=0.8,
        use_judge=("semantic_arg",),
        extra={"acme.ticket": "T-1", "nested": {"a": [1, None, True]}},
    )


def rich_case_1b() -> AgentTestCase:
    """A case that uses every phase 1B matcher, field and sub-block."""
    return AgentTestCase(
        "refund_1b",
        "I want a refund for my broken headphones",
        follow_ups=("It's A12",),
        fakes=(
            FakeSpec("get_order", effect="read"),
            FakeSpec("check_policy", effect="read"),
            FakeSpec("issue_refund"),
            FakeSpec("billing_agent", kind="subagent", effect="read"),
        ),
        expect=Expect(
            decision=Decision.ASK,
            decision_by_turn={1: Decision.CALL},
            tools=(
                ToolExpect(
                    "get_order",
                    {
                        "order_id": Normalized(["A-12", "a12"], strip_chars="-"),
                        "items": Unordered(["a", Regex("^b"), 3]),
                        "tags": Contains(["fragile", Eq("x")]),
                        "address": DictMatch(
                            {
                                "city": Semantic("New York", rubric="same city"),
                                "zip": TypeIs("string"),
                            },
                            mode="exact",
                            ignore=("geo_*",),
                            defaults={"country": "US"},
                        ),
                        "date": DateEq("2026-11-11T10:00:00Z", tz="Europe/Berlin"),
                        "code": Predicate("is_valid_code"),
                        "n": TypeIs(["integer", "number"]),
                    },
                    id="lookup",
                    ignore=("trace_*", "request_id"),
                    weight=2.5,
                ),
                ToolExpect("check_policy", id="policy", same_step_as="lookup"),
                ToolExpect("issue_refund", id="refund", after=("lookup",), turn=1),
            ),
            order="partial",
            forbidden=(ForbiddenCall("issue_refund", unless_after="lookup"),),
            max_llm_calls=4,
            routes=RouteExpect(path=("billing_agent",), mode="first"),
            loops=LoopExpect(max_identical_calls=None, key="name", max_retries_after_error=0),
            intent=("billing", "refunds"),
            intent_labels=("billing", "refunds", "shipping"),
            intent_source="call:get_order.args.topic",
            clarify=ClarifyExpect(
                must_mention_any=("order id",),
                must_mention_all=("order",),
                max_questions=1,
                allow_read_calls_before=False,
                judge_relevance=True,
            ),
            refuse=RefuseExpect(must_mention_any=("not eligible",), offer_alternative=True),
            scope=ScopeExpect(True, topics=("refunds",)),
            plan=PlanExpect(
                optimal_tool_calls=2, constraints=("refund_after_lookup",), judge_quality=True
            ),
            answer_quality=AnswerQualityExpect(
                "Refund issued", accept="partially_correct", criteria=("amount",)
            ),
            grounded=GroundednessExpect(
                sources=("get_*",), include_user_statements=False, allow_unsupported=True
            ),
        ),
        use_judge=("semantic_arg", "answer_quality"),
    )


def rich_run() -> AgentRun:
    return AgentRun(
        case_id="refund_rich",
        input="Refund order A12",
        follow_ups=("thanks",),
        steps=(
            Step(
                1,
                "model",
                "fake-llm",
                args={
                    "messages": [{"role": "user", "content": "Refund order A12"}],
                    "tools": ["get_order"],
                },
                result={"role": "assistant", "content": ""},
                ts=0.5,
                duration_ms=1.25,
                usage=Usage(input_tokens=10, output_tokens=5, latency_s=0.001, cost_usd=0.0001),
            ),
            Step(
                2,
                "tool",
                "get_order",
                args={"order_id": "A12"},
                result={"order_id": "A12", "total": 59.99},
                effect="read",
                ts=2.0,
                duration_ms=0.5,
            ),
            Step(
                3,
                "tool",
                "issue_refund",
                args={"order_id": "Z999"},
                error="FakeToolError: order service down",
                effect="write",
                ts=3.0,
                duration_ms=0.25,
                turn=1,
                parallel_group=2,
                lossy=True,
                provider_call_id="call_abc",
                extra={"acme.k": 1},
            ),
            Step(
                4,
                "subagent",
                "faq_agent",
                args={"input": "hi"},
                result="see FAQ",
                effect="read",
                ts=4.0,
            ),
        ),
        final_answer={"answer": "Refunded", "n": [1, 2]},
        final_text='{"answer": "Refunded", "n": [1, 2]}',
        turn_outputs=("ask", {"answer": "Refunded", "n": [1, 2]}),
        messages=(Message("user", "Refund order A12"), Message("assistant", "Refunded")),
        available_tools=(
            ToolInfo("get_order", "tool", "read", description="Look up", schema={"type": "object"}),
            ToolInfo("faq_agent", "subagent", "read"),
        ),
        model_steps_captured=True,
        started_at="2026-11-12T10:00:00Z",
        usage=Usage(input_tokens=10, output_tokens=5),
        extra={"acme.run": "r1"},
    )


def rich_result() -> RunResult:
    return RunResult(
        "crashed",
        rich_run(),
        error=ErrorInfo("ValueError", "builtins", "ValueError: boom", traceback="...(truncated)"),
        violations=(
            Violation(
                "unexpected_call",
                "get_order({}) matched no response",
                fake="get_order",
                call_id="call-2",
            ),
        ),
        warnings=("agent returned None",),
        trace_id="3f2a9c1d",
        duration_ms=12.5,
        setup_ms=0.5,
        extra={"acme.r": True},
    )


def every_model_instance() -> list[object]:
    case = rich_case()
    case_1b = rich_case_1b()
    return [
        case_1b,
        case_1b.expect.routes,
        case_1b.expect.loops,
        case_1b.expect.plan,
        case_1b.expect.clarify,
        case_1b.expect.refuse,
        case_1b.expect.scope,
        case_1b.expect.answer_quality,
        case_1b.expect.grounded,
        case_1b.expect.tools[0].args["order_id"],
        case_1b.expect.tools[0].args["items"],
        case_1b.expect.tools[0].args["tags"],
        case_1b.expect.tools[0].args["address"],
        case_1b.expect.tools[0].args["date"],
        case_1b.expect.tools[0].args["code"],
        case_1b.expect.tools[0].args["n"],
        case_1b.expect.tools[0].args["address"].fields["city"],
        case,
        case.expect,
        case.expect.tools[0],
        case.expect.forbidden[1],
        case.fakes[0],
        case.fakes[0].responses[0],
        case.input[2],
        case.input[2].tool_calls[0],
        rich_run(),
        rich_run().steps[1],
        rich_run().steps[0].usage,
        rich_result(),
        rich_result().error,
        rich_result().violations[0],
        rich_run().available_tools[0],
        StepEvent(
            "end", "t", 2, "call-2", "tool", "get_order", 0, result={"a": 1}, duration_ms=1.0
        ),
        Verdict(
            "tool_order",
            False,
            0.5,
            "expected a before b",
            step_index=2,
            scorer="judge",
            extra={"k": 1},
        ),
        Eq(1),
        Eq({"a": 1}),
        Eq(1, coerce_numeric=False),
        Eq({"$m": "x"}),
        OneOf([1, "a"]),
        AnyValue(),
        Absent(),
        Regex("^a", flags=2),
        Approx(1.5, abs_tol=0.1),
        Ref("call:lookup.result.order_id"),
        Optional_(Ref("input")),
    ]
