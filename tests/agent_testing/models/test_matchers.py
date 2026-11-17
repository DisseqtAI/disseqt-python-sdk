"""Matcher data classes, the Ref grammar and the matcher registry."""

import dataclasses
from dataclasses import dataclass

import pytest

from disseqt_agent_testing.models import _registry
from disseqt_agent_testing.models._frozen import FrozenDict
from disseqt_agent_testing.models.errors import ModelValidationError
from disseqt_agent_testing.models.matchers import (
    Absent,
    AnyValue,
    Approx,
    Contains,
    DateEq,
    DictMatch,
    Eq,
    Matcher,
    Normalized,
    OneOf,
    Optional_,
    Predicate,
    Ref,
    Regex,
    Semantic,
    TypeIs,
    Unordered,
    matcher_map,
    parse_ref_source,
    reject_undecidable,
    to_matcher,
    walk_matchers,
)


class TestBasicMatchers:
    def test_eq_freezes_and_defaults(self):
        eq = Eq({"a": [1, 2]})
        assert eq.coerce_numeric is True
        assert isinstance(eq.value, FrozenDict) and eq.value["a"] == (1, 2)
        hash(eq)

    def test_eq_coerce_numeric_is_keyword_only(self):
        assert Eq(1, coerce_numeric=False).coerce_numeric is False
        with pytest.raises(TypeError):
            Eq(1, False)  # type: ignore[misc]

    def test_eq_rejects_non_json_and_matchers(self):
        with pytest.raises(ModelValidationError, match="Eq.value"):
            Eq(object())
        with pytest.raises(ModelValidationError, match="not a matcher"):
            Eq(Eq(1))

    def test_one_of_requires_values(self):
        assert OneOf(["a", 1]).values == ("a", 1)
        with pytest.raises(ModelValidationError, match="OneOf.values"):
            OneOf([])
        with pytest.raises(ModelValidationError, match="OneOf.values"):
            OneOf("abc")  # type: ignore[arg-type]

    def test_any_and_absent_have_no_fields(self):
        assert dataclasses.fields(AnyValue()) == () and dataclasses.fields(Absent()) == ()
        assert AnyValue() == AnyValue() and AnyValue() != Absent()

    def test_regex_validates_at_construction(self):
        assert Regex("^2026-").flags == 0
        with pytest.raises(ModelValidationError, match="Regex.pattern: invalid regular expression"):
            Regex("(")
        with pytest.raises(ModelValidationError, match="Regex.flags"):
            Regex("a", flags=-1)

    @pytest.mark.parametrize(
        "kwargs",
        [{"abs_tol": -1}, {"rel_tol": -1e-3}, {"abs_tol": float("nan")}, {"rel_tol": float("inf")}],
    )
    def test_approx_rejects_bad_tolerances(self, kwargs):
        with pytest.raises(ModelValidationError, match="Approx"):
            Approx(1.0, **kwargs)

    def test_approx_value_must_be_a_finite_number(self):
        assert Approx(59).value == 59.0 and isinstance(Approx(59).value, float)
        for bad in (True, "1", float("nan"), None):
            with pytest.raises(ModelValidationError, match="Approx.value"):
                Approx(bad)  # type: ignore[arg-type]

    def test_immutable(self):
        eq = Eq(1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            eq.value = 2  # type: ignore[misc]


class TestRef:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("call:lookup.result.order_id", ("call", "lookup", ("result", "order_id"))),
            ("call:call-3.args.items[0].id", ("call", "call-3", ("args", "items", 0, "id"))),
            ("call:search_*.result", ("call", "search_*", ("result",))),
            ("call:a.result[2][0]", ("call", "a", ("result", 2, 0))),
            ("input", ("input", None, ())),
            ("input:1", ("input", "1", ())),
            ("fake:get_order.total", ("fake", "get_order", ("total",))),
            ("fake:get_order", ("fake", "get_order", ())),
        ],
    )
    def test_valid_sources(self, source, expected):
        assert parse_ref_source(source) == expected
        assert Ref(source).source == source

    @pytest.mark.parametrize(
        ("source", "position", "problem"),
        [
            ("", None, "must be a non-empty string"),
            ("lookup.result.id", 0, "must start with"),
            ("call:", 5, "expected a call id"),
            ("call:lookup", 11, "expected '.result' or '.args'"),
            ("call:lookup.value.id", 11, "expected '.result' or '.args'"),
            ("call:lookup.resulty", 18, "unexpected character"),
            ("call:lookup.result.", 19, "expected a key after '.'"),
            ("call:lookup.result[x]", 19, "expected an index"),
            ("call:lookup.result[0", 19, "expected an index"),
            ("call:lookup.result id", 18, "unexpected character"),
            ("input:abc", 6, "expected a turn number"),
            ("fake:", 5, "expected a fake name"),
        ],
    )
    def test_invalid_sources_name_the_position(self, source, position, problem):
        with pytest.raises(ModelValidationError) as err:
            Ref(source)
        message = str(err.value)
        assert problem in message
        if position is not None:
            assert f"at position {position}" in message

    def test_ref_is_hashable_and_equal_by_source(self):
        assert Ref("input") == Ref("input") and hash(Ref("input")) == hash(Ref("input"))


class TestOptional:
    def test_wraps_plain_values(self):
        assert Optional_("cm").inner == Eq("cm")
        assert Optional_(Regex("^a")).inner == Regex("^a")

    @pytest.mark.parametrize("inner", [Optional_(Eq(1)), Absent()])
    def test_rejects_nonsense_inner(self, inner):
        with pytest.raises(ModelValidationError, match="cannot be wrapped"):
            Optional_(inner)


class TestHelpers:
    def test_to_matcher(self):
        assert to_matcher("A12") == Eq("A12")
        assert to_matcher(Regex("a")) == Regex("a")
        with pytest.raises(ModelValidationError, match="ToolExpect.args.x"):
            to_matcher(object(), "ToolExpect.args.x")

    def test_matcher_map(self):
        result = matcher_map({"a": 1, "b": Absent()}, "ToolExpect.args")
        assert isinstance(result, FrozenDict)
        assert result == {"a": Eq(1), "b": Absent()}
        assert matcher_map(None, "x") == {}
        with pytest.raises(ModelValidationError, match="argument names"):
            matcher_map({"": 1}, "ToolExpect.args")
        with pytest.raises(ModelValidationError, match="object of argument matchers"):
            matcher_map([1], "ToolExpect.args")

    def test_walk_finds_nested_matchers(self):
        nested = {"a": Optional_(Ref("input")), "b": [Eq(1)]}
        kinds = [type(m).__name__ for m in walk_matchers(nested)]
        assert kinds == ["Optional_", "Ref", "Eq"] or sorted(kinds) == ["Eq", "Optional_", "Ref"]

    def test_reject_undecidable(self):
        reject_undecidable({"a": Eq(1)}, "ForbiddenCall.args", "ForbiddenCall.args")
        with pytest.raises(ModelValidationError, match="matcher Ref is not allowed in"):
            reject_undecidable(
                {"a": Optional_(Ref("input"))}, "ForbiddenCall.args", "ForbiddenCall.args"
            )


class TestRegistry:
    def test_built_in_tags(self):
        assert set(_registry.matcher_tags()) >= {
            "eq", "one_of", "any", "absent", "regex", "approx", "ref", "optional",
        }  # fmt: skip

    def test_register_and_unregister_temporary_matcher(self):
        @dataclass(slots=True, frozen=True)
        class Always(Matcher):
            pass

        _registry.register_matcher("always", Always)
        try:
            assert _registry.matcher_class("always") is Always
            assert _registry.matcher_tag(Always) == "always"
            with pytest.raises(ModelValidationError, match="already registered"):
                _registry.register_matcher("always", type("Other", (), {}))
        finally:
            _registry.unregister_matcher("always")
        assert "always" not in _registry.matcher_tags()

    def test_unknown_tag_lists_known_tags(self):
        with pytest.raises(ModelValidationError, match=r"unknown matcher tag 'nope'; known tags"):
            _registry.matcher_class("nope")

    def test_bad_tag_rejected(self):
        with pytest.raises(ModelValidationError, match="must be letters"):
            _registry.register_matcher("bad tag", Eq)


class TestPhase1BMatchers:
    def test_normalized(self):
        assert Normalized("A-123").values == ("A-123",)
        assert Normalized(["a", "b"], strip_chars="-", casefold=False).strip_chars == "-"
        for bad in ([], "", [1], None):
            with pytest.raises(ModelValidationError, match="Normalized.values"):
                Normalized(bad)  # type: ignore[arg-type]

    def test_type_is(self):
        assert TypeIs("integer").types == ("integer",)
        assert TypeIs(["integer", "number"]).types == ("integer", "number")
        for bad in ("int", ["string", "float"], []):
            with pytest.raises(ModelValidationError, match="TypeIs.types"):
                TypeIs(bad)

    def test_unordered_and_contains_wrap_plain_items(self):
        assert Unordered(["a", Regex("^b")]).items == (Eq("a"), Regex("^b"))
        assert Contains(["x", 1]).items == (Eq("x"), Eq(1))
        assert Unordered([]).items == ()  # an empty list is a valid "must be empty" matcher
        with pytest.raises(ModelValidationError, match="Contains.items: must not be empty"):
            Contains([])
        with pytest.raises(ModelValidationError, match=r"Unordered.items\[0\]"):
            Unordered([object()])
        with pytest.raises(ModelValidationError, match="Unordered.items"):
            Unordered("abc")  # type: ignore[arg-type]

    def test_dict_match(self):
        match = DictMatch(
            {"a": 1, "b": Regex("x")}, mode="exact", ignore=["id_*"], defaults={"c": 0}
        )
        assert match.fields == {"a": Eq(1), "b": Regex("x")}
        assert match.ignore == ("id_*",) and match.defaults == {"c": Eq(0)}
        assert DictMatch().fields == {} and DictMatch().mode == "subset"
        with pytest.raises(ModelValidationError, match="DictMatch.mode"):
            DictMatch({}, mode="loose")

    @pytest.mark.parametrize(
        "value",
        ["2026-11-11", "2026-11-11T10:00:00", "2026-11-11T10:00:00Z", "2026-11-11T10:00:00+01:00"],
    )
    def test_date_eq_accepts_iso_values(self, value):
        assert DateEq(value).value == value

    @pytest.mark.parametrize("value", ["next friday", "11/11/2026", "", "2026-13-40"])
    def test_date_eq_rejects_non_iso(self, value):
        with pytest.raises(ModelValidationError, match="DateEq.value"):
            DateEq(value)

    def test_date_eq_zone_name_is_checked_for_shape_only(self):
        assert DateEq("2026-11-11", tz="Europe/Berlin").tz == "Europe/Berlin"
        assert DateEq("2026-11-11", tz="Not/AZone").tz == "Not/AZone"  # resolved at scoring time
        with pytest.raises(ModelValidationError, match="DateEq.tz"):
            DateEq("2026-11-11", tz="Europe Berlin")

    def test_predicate_is_a_name_not_a_path_to_import(self):
        assert Predicate("is_valid_iban").name == "is_valid_iban"
        assert Predicate("acme.checks:is_even").name == "acme.checks:is_even"  # still just a name
        for bad in ("", "1abc", "has space", "a;b"):
            with pytest.raises(ModelValidationError, match="Predicate.name"):
                Predicate(bad)

    def test_semantic(self):
        assert Semantic("New York", rubric="same city").rubric == "same city"
        assert Semantic({"city": "NYC"}).reference == {"city": "NYC"}
        with pytest.raises(ModelValidationError, match="not a matcher"):
            Semantic(Eq(1))
        with pytest.raises(ModelValidationError, match="Semantic.reference"):
            Semantic(object())

    def test_every_matcher_has_a_distinct_tag(self):
        tags = _registry.matcher_tags()
        assert set(tags) >= {
            "normalized", "type_is", "unordered", "contains", "dict_match", "date_eq",
            "predicate", "semantic",
        }  # fmt: skip
        assert len(tags) == len(set(tags))

    def test_scoring_time_matchers_are_found_when_nested(self):
        nested = {"a": DictMatch({"b": Unordered([Semantic("x")])})}
        assert [type(m).__name__ for m in walk_matchers(nested)].count("Semantic") == 1
        for matcher in (Semantic("x"), Predicate("p"), Ref("input")):
            with pytest.raises(ModelValidationError, match="is not allowed in"):
                reject_undecidable({"k": DictMatch({"v": matcher})}, "ForbiddenCall.args", "x")


class TestHostileInput:
    def test_regex_flag_problems_are_validation_errors_not_raw_exceptions(self):
        import re

        with pytest.raises(ModelValidationError, match="Regex.flags"):
            Regex("a", flags=re.LOCALE)  # not allowed with a str pattern
        with pytest.raises(ModelValidationError, match="Regex.flags"):
            Regex("a", flags=2**70)

    @pytest.mark.parametrize("name", ["ok\n", "ok\n\n"])
    def test_names_with_a_trailing_newline_are_rejected(self, name):
        # ``$`` matches before a trailing newline, so patterns must use fullmatch.
        with pytest.raises(ModelValidationError, match="Predicate.name"):
            Predicate(name)
        with pytest.raises(ModelValidationError, match="DateEq.tz"):
            DateEq("2026-11-11", tz="Europe/Berlin\n")

    def test_a_matcher_registered_at_runtime_serializes_without_serde_changes(self):
        from disseqt_agent_testing.models import from_dict, to_dict

        @dataclass(slots=True, frozen=True)
        class StartsWith(Matcher):
            prefix: str

        _registry.register_matcher("starts_with", StartsWith)
        try:
            from disseqt_agent_testing.models import ToolExpect

            tool = ToolExpect("x", {"a": StartsWith("ab")})
            data = to_dict(tool)
            assert data["args"]["a"] == {"$m": "starts_with", "prefix": "ab"}
            assert from_dict(ToolExpect, data) == tool
        finally:
            _registry.unregister_matcher("starts_with")
        with pytest.raises(ModelValidationError, match="unknown matcher tag 'starts_with'"):
            from disseqt_agent_testing.models import from_dict as load

            load(ToolExpect, {"name": "x", "args": {"a": {"$m": "starts_with", "prefix": "ab"}}})
