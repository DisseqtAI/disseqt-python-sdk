"""Regression tests for the model fix pass (schema, decisions, dates, ids, input escapes)."""

import copy
import datetime as dt

import pytest

from disseqt_agent_testing.models import (
    AgentTestCase,
    Approx,
    DateEq,
    Eq,
    Expect,
    FakeResponse,
    FakeSpec,
    ModelValidationError,
    ToolExpect,
    _migrate,
)
from disseqt_agent_testing.models._serde import from_dict
from disseqt_agent_testing.models.matchers import parse_iso, parse_ref_source


def case(expect, fakes=(), follow_ups=()):
    return AgentTestCase("c", "hi", expect=expect, fakes=fakes, follow_ups=follow_ups)


WRITE = (FakeSpec("issue_refund"), FakeSpec("lookup", effect="read"))


class TestFakeSchema:
    @pytest.mark.parametrize(
        "schema, match",
        [
            ({"properties": ["a"]}, "properties"),
            ({"properties": {"a": 1}}, r"properties\.a"),
            ({"properties": {"a": {"type": "string"}}, "required": "abc"}, "required"),
            ({"properties": {"a": {}}, "required": ["b"]}, "required"),
            ({"properties": {"a": {}}, "required": [1]}, "required"),
        ],
    )
    def test_bad_schema_is_rejected_at_construction(self, schema, match):
        with pytest.raises(ModelValidationError, match=f"FakeSpec.schema.*{match}"):
            FakeSpec("x", schema=schema)

    def test_good_schema(self):
        FakeSpec("x", schema={"properties": {"a": {"type": "string"}}, "required": ["a"]})
        FakeSpec("x", schema={"type": "object", "required": []})


class TestWriteToolVsDecision:
    def test_any_acceptable_call_decision_allows_the_write(self):
        case(Expect(decision=("call", "ask"), tools=(ToolExpect("issue_refund"),)), WRITE)

    def test_tool_without_turn_needs_a_conflict_on_every_turn(self):
        expect = Expect(
            decision="call", decision_by_turn={1: "ask"}, tools=(ToolExpect("issue_refund"),)
        )
        case(expect, WRITE, follow_ups=("x",))

    def test_unconstrained_turn_allows_it(self):
        expect = Expect(decision="ask", tools=(ToolExpect("issue_refund"),))
        case(expect, WRITE, follow_ups=("x",))  # turn 1 has no decision

    def test_genuinely_invalid_still_rejected(self):
        expect = Expect(decision="ask", tools=(ToolExpect("issue_refund"),))
        with pytest.raises(ModelValidationError, match="cannot expect write tool"):
            case(expect, WRITE)
        both = Expect(
            decision="ask", decision_by_turn={1: "refuse"}, tools=(ToolExpect("issue_refund"),)
        )
        with pytest.raises(ModelValidationError, match="cannot expect write tool"):
            case(both, WRITE, follow_ups=("x",))
        pinned = Expect(decision=("ask", "refuse"), tools=(ToolExpect("issue_refund", turn=0),))
        with pytest.raises(ModelValidationError, match="cannot expect write tool"):
            case(pinned, WRITE)


class TestDateSubset:
    @pytest.mark.parametrize(
        "value",
        ["20240101", "2024-01-01T10:00:00.5Z", "2024-01-01 10:00", "2024-W01-1", "2024-1-1"],
    )
    def test_value_set_is_the_documented_subset(self, value):
        if value == "2024-01-01T10:00:00.5Z":
            assert DateEq(value).value == value
        else:
            with pytest.raises(ModelValidationError, match="DateEq.value"):
                DateEq(value)

    def test_parse_iso_is_explicit(self):
        assert parse_iso("2024-01-01T10:00:00.5Z") == dt.datetime(
            2024, 1, 1, 10, 0, 0, 500000, tzinfo=dt.timezone.utc
        )
        assert parse_iso("2024-01-01T10:00+05:30").utcoffset() == dt.timedelta(hours=5, minutes=30)
        assert parse_iso("2024-01-01") == dt.datetime(2024, 1, 1)
        assert parse_iso("2024-01-01T10:00:00.123456").microsecond == 123456
        for bad in ("20240101", "2024-01-01T10:00:00.1234567", "2024-01-01T24:00", "2024-02-30"):
            with pytest.raises(ValueError):
                parse_iso(bad)

    def test_match_time_rejects_what_construction_rejects(self):
        from disseqt_agent_testing.harness._when import matches

        assert not matches(DateEq("2024-01-01"), "20240101")
        assert matches(DateEq("2024-01-01"), "2024-01-01T23:00:00.5Z")


class TestFillIds:
    def test_explicit_ids_are_not_duplicated(self):
        from disseqt_agent_testing.models import Message

        data = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "a"}, {"name": "b", "id": "call_1"}, {"name": "c"}],
        }
        msg = from_dict(Message, data, fill_ids=True)
        ids = [c.id for c in msg.tool_calls]
        assert len(set(ids)) == 3 and ids[1] == "call_1"


class TestInputEscapes:
    def test_huge_approx(self):
        with pytest.raises(ModelValidationError, match="Approx.value"):
            Approx(10**400)
        with pytest.raises(ModelValidationError, match="Approx.abs_tol"):
            Approx(1, abs_tol=10**400)

    @pytest.mark.parametrize("key", ["²", "٣", "１"])
    def test_non_ascii_digits_in_decision_by_turn(self, key):
        with pytest.raises(ModelValidationError, match="decision_by_turn"):
            Expect(decision_by_turn={key: "ask"})

    def test_non_ascii_digit_ref(self):
        with pytest.raises(ModelValidationError, match="Ref.source"):
            parse_ref_source("input:²")

    def test_colliding_turn_keys(self):
        with pytest.raises(ModelValidationError, match="decision_by_turn.*twice|duplicate"):
            Expect(decision_by_turn={1: "ask", "01": "call"})

    def test_huge_int_is_rejected(self):
        with pytest.raises(ModelValidationError, match="Eq.value"):
            Eq(10**5000)
        assert Eq(10**3999).value == 10**3999

    def test_latency_bound(self):
        FakeResponse(latency_s=3600)
        with pytest.raises(ModelValidationError, match="latency_s"):
            FakeResponse(latency_s=3601)


class TestMigrateIsolation:
    def test_migration_cannot_mutate_the_callers_document(self):
        def mutate(doc):
            doc["nested"]["x"] = 2
            return doc

        _migrate.register_migration("Doc", 1, mutate)
        try:
            original = {"schema_version": 1, "nested": {"x": 1}}
            snapshot = copy.deepcopy(original)
            out = _migrate.upgrade("Doc", original, 2)
            assert original == snapshot
            assert out["nested"]["x"] == 2
        finally:
            _migrate.unregister_migration("Doc", 1)
