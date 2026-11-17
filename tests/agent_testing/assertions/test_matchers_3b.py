"""Phase 3b matchers (M-10 ... M-30)."""

from typing import Any

import pytest

from disseqt_agent_testing.assertions import (
    EvalConfig,
    ExpectConfigError,
    JudgeVerdict,
    evaluate,
)
from disseqt_agent_testing.assertions.matching import MatchContext, match_value
from disseqt_agent_testing.assertions.predicates import (
    lookup_predicate,
    register_predicate,
    unregister_predicate,
)
from disseqt_agent_testing.models import (
    Contains,
    DateEq,
    DictMatch,
    Eq,
    Expect,
    Normalized,
    Predicate,
    Semantic,
    ToolExpect,
    TypeIs,
    Unordered,
    freeze_json,
)

from .helpers import run_of, step

CTX = MatchContext()


def one(matcher: Any, actual: Any, ctx: MatchContext = CTX) -> Any:
    out = match_value(matcher, freeze_json(actual), ctx)
    assert len(out) == 1
    return out[0]


def test_m10_m11_normalized_ignores_case_and_punctuation_and_is_lossy_by_design() -> None:
    m = Normalized(("new york",))
    assert one(m, "New-York").ok and one(m, "NEW, YORK.").ok
    assert one(m, "newyork").ok  # documented: deleting separators merges words
    assert one(m, "Boston").code == "value_error:normalized"
    assert one(m, 5).code == "type_error:string->integer"


def test_normalized_options() -> None:
    assert one(Normalized(("A.B",), casefold=False, strip_chars="."), "AB").ok
    assert one(Normalized(("a.b",), casefold=False, strip_chars="."), "AB").ok is False


def test_m14_m15_type_is_bool_int_number_rules() -> None:
    assert (
        one(TypeIs(("integer",)), 3).ok
        and one(TypeIs(("number",)), 3).ok
        and one(TypeIs(("number",)), 3.5).ok
    )
    assert (
        one(TypeIs(("integer",)), True).ok is False and one(TypeIs(("number",)), True).ok is False
    )
    assert one(TypeIs(("integer",)), 3.5).ok is False
    assert one(TypeIs(("boolean",)), True).ok and one(TypeIs(("null",)), None).ok
    r = one(TypeIs(("string", "array")), 5)
    assert r.code == "type_error:string|array->integer"


def test_m16_unordered_matches_dict_elements_and_reports_missing_and_extra() -> None:
    m = Unordered(({"id": 1}, {"id": 2}))
    assert one(m, [{"id": 2}, {"id": 1}]).ok
    bad = one(m, [{"id": 1}, {"id": 9}])
    assert bad.code == "value_error:unordered"
    assert list(bad.details["missing"]) == ['Eq({"id":2})'] and len(bad.details["extra"]) == 1
    assert one(m, [{"id": 1}]).ok is False and one(m, [{"id": 1}, {"id": 2}, {"id": 3}]).ok is False
    assert one(m, "no").code == "type_error:array->string"


def test_unordered_is_a_multiset_not_a_set() -> None:
    assert one(Unordered((1, 1)), [1, 1]).ok
    assert one(Unordered((1, 1)), [1, 2]).ok is False
    assert one(Unordered((1, 2)), [1, 1]).ok is False


def test_m17_contains_for_strings_and_lists() -> None:
    assert one(Contains(("Paris", "rome")), "I flew from PARIS to Rome").ok
    miss = one(Contains(("Paris", "oslo")), "Paris only")
    assert miss.code == "value_error:contains" and list(miss.details["missing"]) == ["oslo"]
    assert one(Contains((1, 2)), [3, 2, 1]).ok
    assert one(Contains((1, 1)), [1, 2]).ok is False  # distinct elements
    assert one(Contains((1,)), 5).code == "type_error:array->integer"


def test_m18_dict_match_modes() -> None:
    sub = DictMatch({"a": 1})
    assert one(sub, {"a": 1, "b": 2}).ok
    exact = DictMatch({"a": 1}, mode="exact")
    out = match_value(exact, freeze_json({"a": 1, "b": 2}), CTX, ("arg",))
    assert [(r.path, r.code) for r in out if not r.ok] == [(("arg", "b"), "unexpected_param")]
    assert one(sub, [1]).code == "type_error:object->array"


def test_m19_m20_dict_match_ignore_globs_are_case_sensitive() -> None:
    m = DictMatch({"a": 1}, mode="exact", ignore=("_*", "Meta"))
    assert all(r.ok for r in match_value(m, freeze_json({"a": 1, "_id": 5, "Meta": 1}), CTX))
    assert not all(r.ok for r in match_value(m, freeze_json({"a": 1, "meta": 1}), CTX))


def test_m21_dict_match_defaults_are_stripped_from_the_actual() -> None:
    m = DictMatch({"a": 1}, mode="exact", defaults={"page": 1})
    assert all(r.ok for r in match_value(m, freeze_json({"a": 1, "page": 1}), CTX))
    assert not all(r.ok for r in match_value(m, freeze_json({"a": 1, "page": 2}), CTX))


def test_m24_date_eq_instants_across_offsets() -> None:
    m = DateEq("2026-03-01T12:00:00Z")
    assert one(m, "2026-03-01T13:00:00+01:00").ok and one(m, "2026-03-01T12:00:00Z").ok
    assert one(m, "2026-03-01T12:00:01Z").code == "value_error:date"


def test_m25_date_only_compares_dates_in_a_zone() -> None:
    m = DateEq("2026-03-01")
    assert one(m, "2026-03-01").ok and one(m, "2026-03-01T23:59:00Z").ok
    assert one(m, "2026-03-02").ok is False
    tokyo = DateEq("2026-03-02", tz="Asia/Tokyo")
    assert one(tokyo, "2026-03-01T20:00:00Z").ok  # 05:00 next day in Tokyo


def test_m26_naive_vs_aware_needs_a_zone() -> None:
    m = DateEq("2026-03-01T12:00:00Z")
    assert one(m, "2026-03-01T12:00:00").code == "type_error:naive_aware"
    assert one(DateEq("2026-03-01T12:00:00", tz="UTC"), "2026-03-01T12:00:00Z").ok


def test_m27_natural_language_dates_fail_with_a_parse_code() -> None:
    assert one(DateEq("2026-03-01"), "next friday").code == "value_error:date_parse"
    assert one(DateEq("2026-03-01"), 5).code == "type_error:string->integer"


def test_a_missing_time_zone_database_is_a_clear_config_error() -> None:
    with pytest.raises(ExpectConfigError, match="tzdata"):
        one(DateEq("2026-03-01", tz="Nowhere/Land"), "2026-03-01")


def test_m28_predicates_by_name() -> None:
    register_predicate("is_even", lambda v: v % 2 == 0)
    register_predicate("with_reason", lambda v: (False, "too small"))
    register_predicate("boom", lambda v: 1 / 0)
    register_predicate("junk", lambda v: "yes")
    try:
        assert (
            one(Predicate("is_even"), 4).ok
            and one(Predicate("is_even"), 3).code == "value_error:predicate"
        )
        assert one(Predicate("with_reason"), 1).reason == "too small"
        boom = one(Predicate("boom"), 1)
        assert (boom.status, boom.code) == (
            "error",
            "predicate_error",
        ) and "ZeroDivisionError" in boom.reason
        assert one(Predicate("junk"), 1).code == "predicate_error"
        with pytest.raises(ValueError, match="already registered"):
            register_predicate("is_even", lambda v: True)
    finally:
        for n in ("is_even", "with_reason", "boom", "junk"):
            unregister_predicate(n)


def test_m29_unknown_predicate_names_are_config_errors_before_scoring() -> None:
    expect = Expect(tools=(ToolExpect("t", {"a": Predicate("nope")}),))
    with pytest.raises(ExpectConfigError, match="unknown predicate 'nope'"):
        evaluate(run_of(step(1, "t", {"a": 1})), expect)
    cfg = EvalConfig(predicates={"nope": lambda v: v == 1})
    assert evaluate(run_of(step(1, "t", {"a": 1})), expect, config=cfg).outcome == "passed"
    assert lookup_predicate("nope", cfg.predicates)(1) is True


class Judge:
    def __init__(self, label: str = "equivalent", status: str = "ok") -> None:
        self.label, self.status, self.tasks = label, status, []

    def judge(self, task: Any) -> JudgeVerdict:
        self.tasks.append(task)
        return JudgeVerdict(self.status, self.label)  # type: ignore[arg-type]

    async def ajudge(self, task: Any) -> JudgeVerdict:
        return self.judge(task)


def test_m30_semantic_without_a_judge_is_skipped_and_the_outcome_incomplete() -> None:
    r = one(Semantic("New York City"), "NYC")
    assert (r.status, r.code, r.score) == ("skipped", "judge_unavailable", 1.0)
    expect = Expect(tools=(ToolExpect("t", {"city": Semantic("New York City")}),))
    rep = evaluate(run_of(step(1, "t", {"city": "NYC"})), expect)
    assert rep.check("tool_args").reason_code == "judge_unavailable" and rep.outcome == "incomplete"  # type: ignore[union-attr]
    assert rep.check("tool_args").details["partial_value"] == 1.0  # type: ignore[union-attr]


def test_m30_a_judge_decides_and_marks_the_result_judged() -> None:
    expect = Expect(tools=(ToolExpect("t", {"city": Semantic("New York City")}),))
    run = run_of(step(1, "t", {"city": "Boston"}))
    bad = evaluate(run, expect, judge=Judge("not_equivalent"), use_judge=True)
    got = bad.check("tool_args")
    assert got.passed is False and got.judged and got.reason_code == "value_error:semantic"  # type: ignore[union-attr]
    good = evaluate(run, expect, judge=Judge("equivalent"), use_judge=True)
    assert good.outcome == "passed" and good.check("tool_args").judged  # type: ignore[union-attr]
    assert good.judge_calls[0]["kind"] == "semantic_arg"
    unsure = evaluate(run, expect, judge=Judge("unsure"), use_judge=True)
    assert unsure.check("tool_args").reason_code == "judge_unsure"  # type: ignore[union-attr]
    broken = evaluate(run, expect, judge=Judge("x", status="error"), use_judge=True)
    assert broken.check("tool_args").status == "error"  # type: ignore[union-attr]


def test_s05_only_matched_pairs_are_judged_never_every_slot_by_call_pair() -> None:
    tools = (
        ToolExpect("t", {"q": Semantic("a")}, id="e1"),
        ToolExpect("t", {"q": Semantic("b")}, id="e2"),
    )
    steps = [step(i, "t", {"q": f"v{i}"}) for i in (1, 2, 3)]
    judge = Judge("equivalent")
    evaluate(run_of(*steps), Expect(tools=tools, extra_calls="allow"), judge=judge, use_judge=True)
    assert len(judge.tasks) == 2  # 2 matched pairs, not 2 x 3


def test_semantic_in_a_veto_is_rejected_at_construction() -> None:
    from disseqt_agent_testing.models import ForbiddenCall, ModelValidationError

    with pytest.raises(ModelValidationError):
        ForbiddenCall("x", {"a": Semantic("b")})


def test_eq_is_reflexive_on_random_json_and_unordered_ignores_permutations() -> None:
    import random

    rng = random.Random(3)

    def gen(depth: int = 0) -> Any:
        kind = rng.choice(
            ["int", "float", "str", "bool", "none", "list", "dict"]
            if depth < 3
            else ["int", "str", "bool"]
        )
        return {
            "int": lambda: rng.randint(-5, 5),
            "float": lambda: rng.random(),
            "str": lambda: rng.choice(["a", "B", "c d"]),
            "bool": lambda: rng.random() < 0.5,
            "none": lambda: None,
            "list": lambda: [gen(depth + 1) for _ in range(rng.randint(0, 3))],
            "dict": lambda: {rng.choice("xyz"): gen(depth + 1) for _ in range(rng.randint(0, 3))},
        }[kind]()

    for _ in range(500):
        value = freeze_json(gen())
        assert one(Eq(value), value).ok
        items = [gen() for _ in range(rng.randint(0, 4))]
        shuffled = list(items)
        rng.shuffle(shuffled)
        assert one(Unordered(tuple(items)), shuffled).ok
