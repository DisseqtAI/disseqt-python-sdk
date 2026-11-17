"""to_dict / from_dict: round trips, strict decoding, matcher tagging, versions, policies."""

import json
import sys

import pytest

from disseqt_agent_testing.models import (
    AgentRun,
    AgentTestCase,
    Approx,
    Decision,
    Eq,
    Expect,
    FrozenDict,
    Message,
    ModelValidationError,
    MsgToolCall,
    Optional_,
    Ref,
    RunResult,
    ToolExpect,
    _migrate,
    from_dict,
    to_dict,
)

from .factories import every_model_instance, rich_case, rich_result


def json_round_trip(obj):
    return from_dict(type(obj), json.loads(json.dumps(to_dict(obj), allow_nan=False)))


@pytest.mark.parametrize("obj", every_model_instance(), ids=lambda o: type(o).__name__)
def test_round_trip_every_model(obj):
    assert json_round_trip(obj) == obj


def test_round_trip_keeps_types_and_order():
    again = json_round_trip(rich_case())
    assert isinstance(again.expect.tools, tuple) and isinstance(again.fakes[0].schema, FrozenDict)
    assert again.expect.tools[1].count == (1, 1)
    assert list(again.expect.tools[1].args)[:2] == ["order_id", "amount"]
    assert again.input[2].tool_calls[1].parse_error == "Expecting value"


def test_encoded_output_is_plain_json_types():
    def check(value):
        assert type(value) in (dict, list, str, int, float, bool, type(None)), type(value)
        if isinstance(value, dict):
            [check(v) for v in value.values()]
        if isinstance(value, list):
            [check(v) for v in value]

    check(to_dict(rich_case()))
    check(to_dict(rich_result()))


class TestDecodingIsStrict:
    def good(self):
        return to_dict(AgentTestCase("c", "hi"))

    def test_unknown_field_names_class_and_valid_fields(self):
        data = self.good() | {"argz": 1}
        with pytest.raises(ModelValidationError) as err:
            from_dict(AgentTestCase, data)
        assert "AgentTestCase: unknown field(s) ['argz']; valid: [" in str(err.value)

    def test_unknown_field_in_nested_model_has_a_path(self):
        data = to_dict(AgentTestCase("c", "hi", expect=Expect(tools=(ToolExpect("x"),))))
        data["expect"]["tools"][0]["argz"] = 1
        with pytest.raises(ModelValidationError, match=r"AgentTestCase.expect.tools\[0\]: unknown"):
            from_dict(AgentTestCase, data)

    def test_missing_required_field(self):
        data = self.good()
        del data["id"]
        with pytest.raises(
            ModelValidationError, match="AgentTestCase.id: required field is missing"
        ):
            from_dict(AgentTestCase, data)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("repeats", "3", "expected an int, got str"),
            ("repeats", True, "expected an int, got bool"),
            ("pass_k_threshold", "1", "expected a number"),
            ("tags", "smoke", "expected a list"),
            ("id", 5, "expected a string"),
        ],
    )
    def test_wrong_types(self, field, value, match):
        data = self.good() | {field: value}
        with pytest.raises(ModelValidationError, match=match):
            from_dict(AgentTestCase, data)

    def test_validation_runs_on_load(self):
        data = self.good() | {"repeats": 3, "pass_k": 5}
        with pytest.raises(ModelValidationError, match="pass_k .5. cannot exceed repeats .3."):
            from_dict(AgentTestCase, data)

    def test_int_is_accepted_for_float_fields(self):
        data = to_dict(AgentTestCase("c", "hi", expect=Expect(max_latency_s=2.5)))
        data["expect"]["max_latency_s"] = 5
        assert from_dict(AgentTestCase, data).expect.max_latency_s == 5.0

    def test_not_an_object(self):
        with pytest.raises(ModelValidationError, match="expected an object, got list"):
            from_dict(AgentTestCase, [])  # type: ignore[arg-type]

    def test_literal_and_enum_errors_list_choices(self):
        data = to_dict(AgentTestCase("c", "hi", expect=Expect(order="partial")))
        data["expect"]["order"] = "sorted"
        with pytest.raises(ModelValidationError, match="must be one of"):
            from_dict(AgentTestCase, data)
        data["expect"]["order"] = "any"
        data["expect"]["decision"] = ["maybe"]
        with pytest.raises(ModelValidationError, match=r"expect.decision\[0\]: must be one of"):
            from_dict(AgentTestCase, data)


class TestShorthands:
    def test_string_input_is_accepted_and_a_list_is_always_emitted(self):
        case = from_dict(AgentTestCase, {"id": "c", "input": "hi"})
        assert case.input == (Message("user", "hi"),)
        assert to_dict(case)["input"] == [{"role": "user", "content": "hi"}]

    def test_decision_shorthands(self):
        data = {
            "id": "c",
            "input": "hi",
            "expect": {"decision": "ask", "decision_by_turn": {1: "call"}},
        }
        case = from_dict(AgentTestCase, data)
        assert case.expect.decision == (Decision.ASK,)
        assert case.expect.decision_by_turn == {"1": (Decision.CALL,)}

    def test_count_forms(self):
        for raw, expected in ((2, (2, 2)), ([1, None], (1, None)), (None, None)):
            data = {"id": "c", "input": "hi", "expect": {"tools": [{"name": "x", "count": raw}]}}
            assert from_dict(AgentTestCase, data).expect.tools[0].count == expected
        data = {"id": "c", "input": "hi", "expect": {"tools": [{"name": "x", "count": True}]}}
        with pytest.raises(ModelValidationError, match="count"):
            from_dict(AgentTestCase, data)

    def test_max_tool_calls_forms(self):
        for raw in (3, {"a_*": 2}):
            data = {"id": "c", "input": "hi", "expect": {"max_tool_calls": raw}}
            assert from_dict(AgentTestCase, data).expect.max_tool_calls == raw


class TestMatcherTagging:
    def tools(self, **args):
        return to_dict(Expect(tools=(ToolExpect("x", args),)))["tools"][0]["args"]

    def test_plain_values_use_the_shorthand(self):
        assert self.tools(a="A") == {"a": "A"}
        assert self.tools(a=[1, 2]) == {"a": [1, 2]}
        assert self.tools(a={"k": 1}) == {"a": {"k": 1}}

    def test_eq_that_is_not_lossless_is_tagged(self):
        assert self.tools(a=Eq(1, coerce_numeric=False)) == {
            "a": {"$m": "eq", "value": 1, "coerce_numeric": False}
        }

    def test_dict_containing_the_tag_key_is_wrapped_and_round_trips(self):
        args = self.tools(a=Eq({"$m": "approx"}))
        assert args == {"a": {"$m": "eq", "value": {"$m": "approx"}, "coerce_numeric": True}}
        expect = Expect(tools=(ToolExpect("x", {"a": Eq({"$m": "approx"})}),))
        assert json_round_trip(expect) == expect

    def test_tagged_forms(self):
        assert self.tools(a=Approx(59.99, abs_tol=0.01))["a"] == {
            "$m": "approx", "value": 59.99, "abs_tol": 0.01, "rel_tol": 1e-9,
        }  # fmt: skip
        assert self.tools(a=Optional_(Ref("input")))["a"] == {
            "$m": "optional", "inner": {"$m": "ref", "source": "input"},
        }  # fmt: skip

    def test_decoding_shorthand_and_tags(self):
        data = {"name": "x", "args": {"a": "A", "b": {"$m": "regex", "pattern": "^x"}}}
        tool = from_dict(ToolExpect, data)
        assert tool.args["a"] == Eq("A") and tool.args["b"].pattern == "^x"

    def test_unknown_tag_lists_known_tags(self):
        with pytest.raises(
            ModelValidationError, match=r"unknown matcher tag 'nope'; known tags: \["
        ):
            from_dict(ToolExpect, {"name": "x", "args": {"a": {"$m": "nope"}}})

    def test_unknown_field_inside_a_matcher(self):
        data = {"name": "x", "args": {"a": {"$m": "approx", "value": 1, "tolerance": 2}}}
        with pytest.raises(ModelValidationError, match="ToolExpect.args.a: unknown field"):
            from_dict(ToolExpect, data)

    def test_invalid_ref_is_caught_on_load(self):
        data = {"name": "x", "args": {"a": {"$m": "ref", "source": "call:lookup"}}}
        with pytest.raises(ModelValidationError, match="invalid ref"):
            from_dict(ToolExpect, data)

    def test_answer_contains_accepts_text_and_matchers_only(self):
        data = {"answer_contains": ["ok", {"$m": "regex", "pattern": "5"}]}
        expect = from_dict(Expect, data)
        assert expect.answer_contains[0] == "ok" and expect.answer_contains[1].pattern == "5"
        with pytest.raises(ModelValidationError, match="string or a"):
            from_dict(Expect, {"answer_contains": [3]})


class TestVersions:
    def test_version_is_emitted_first_for_documents_only(self):
        assert list(to_dict(AgentTestCase("c", "hi")))[0] == "schema_version"
        assert to_dict(AgentTestCase("c", "hi"))["schema_version"] == 1
        assert "schema_version" in to_dict(AgentRun()) and "schema_version" in to_dict(
            rich_result()
        )
        assert "schema_version" not in to_dict(Expect())

    def test_missing_version_means_one(self):
        assert from_dict(AgentTestCase, {"id": "c", "input": "hi"}).id == "c"

    def test_current_version_is_accepted(self):
        assert from_dict(AgentTestCase, {"schema_version": 1, "id": "c", "input": "hi"})

    def test_newer_version_is_rejected_with_an_upgrade_hint(self):
        with pytest.raises(
            ModelValidationError, match="schema_version 2, this SDK reads up to 1; upgrade"
        ):
            from_dict(AgentTestCase, {"schema_version": 2, "id": "c", "input": "hi"})

    @pytest.mark.parametrize("bad", ["1", True, -1, 1.5])
    def test_bad_version_values(self, bad):
        with pytest.raises(ModelValidationError, match="schema_version"):
            from_dict(AgentTestCase, {"schema_version": bad, "id": "c", "input": "hi"})

    def test_nested_documents_carry_their_own_version(self):
        encoded = to_dict(rich_result())
        assert encoded["schema_version"] == 1 and encoded["run"]["schema_version"] == 1

    def test_migration_hook_upgrades_older_files(self):
        def v0_to_v1(data):
            data = dict(data)
            data["input"] = data.pop("prompt")  # a field that was renamed in "version 0"
            return data

        _migrate.register_migration("AgentTestCase", 0, v0_to_v1)
        try:
            case = from_dict(AgentTestCase, {"schema_version": 0, "id": "c", "prompt": "hi"})
            assert case.input == (Message("user", "hi"),)
        finally:
            _migrate.unregister_migration("AgentTestCase", 0)
        with pytest.raises(ModelValidationError, match="no migration from schema_version 0"):
            from_dict(AgentTestCase, {"schema_version": 0, "id": "c", "prompt": "hi"})

    def test_duplicate_migration_is_rejected(self):
        _migrate.register_migration("Probe", 0, lambda d: d)
        try:
            with pytest.raises(ModelValidationError, match="already exists"):
                _migrate.register_migration("Probe", 0, lambda d: d)
        finally:
            _migrate.unregister_migration("Probe", 0)


class TestUnknownFieldPolicies:
    def data(self):
        return {"id": "c", "input": "hi", "owner": "me", "ticket": "T-1"}

    def test_error_is_the_default(self):
        with pytest.raises(ModelValidationError, match=r"unknown field\(s\) \['owner', 'ticket'\]"):
            from_dict(AgentTestCase, self.data())

    def test_extra_moves_unknown_keys_into_the_extra_bag(self):
        case = from_dict(AgentTestCase, self.data() | {"extra": {"k": 1}}, unknown="extra")
        assert case.extra == {"k": 1, "owner": "me", "ticket": "T-1"}

    def test_extra_needs_a_model_with_an_extra_bag(self):
        with pytest.raises(ModelValidationError, match="the model has no 'extra'"):
            from_dict(Message, {"role": "user", "content": "x", "bogus": 1}, unknown="extra")

    def test_ignore_drops_keys_and_warns_once_per_model(self):
        warnings: list[str] = []
        data = self.data()
        data["expect"] = {"tools": [{"name": "x", "bogus": 1}]}
        case = from_dict(AgentTestCase, data, unknown="ignore", warnings=warnings)
        assert case.id == "c" and case.extra == {}
        assert warnings == [
            "AgentTestCase: ignored unknown field(s) ['owner', 'ticket']",
            "AgentTestCase.expect.tools[0]: ignored unknown field(s) ['bogus']",
        ]

    def test_bad_policy_name(self):
        with pytest.raises(ValueError, match="unknown must be one of"):
            from_dict(AgentTestCase, self.data(), unknown="lenient")


class TestFillIds:
    def messages(self):
        return {
            "id": "c",
            "input": [
                {"role": "assistant", "tool_calls": [{"name": "a"}, {"id": "keep", "name": "b"}]},
                {"role": "tool", "tool_call_id": "keep"},
                {"role": "assistant", "tool_calls": [{"name": "c"}]},
                {"role": "user", "content": "go"},
            ],
        }

    def test_ids_are_filled_by_position_when_asked(self):
        case = from_dict(AgentTestCase, self.messages(), fill_ids=True)
        ids = [c.id for m in case.input for c in m.tool_calls]
        assert ids == ["call_1", "keep", "call_3"]

    def test_loading_twice_gives_equal_results(self):
        assert from_dict(AgentTestCase, self.messages(), fill_ids=True) == from_dict(
            AgentTestCase, self.messages(), fill_ids=True
        )

    def test_without_the_flag_a_missing_id_is_an_error(self):
        with pytest.raises(
            ModelValidationError, match=r"input\[0\].tool_calls\[0\].id: required field is missing"
        ):
            from_dict(AgentTestCase, self.messages())


class TestMessageJson:
    def test_defaults_are_omitted_but_content_is_always_emitted(self):
        assert to_dict(Message("assistant")) == {"role": "assistant", "content": ""}
        call = MsgToolCall("c", "x")
        assert to_dict(call) == {"id": "c", "name": "x"}

    def test_none_content_loads_as_empty(self):
        assert from_dict(Message, {"role": "assistant", "content": None}).content == ""
        assert from_dict(Message, {"role": "assistant"}).content == ""

    def test_unknown_message_key_is_rejected_with_a_path(self):
        with pytest.raises(ModelValidationError, match=r"AgentTestCase.input\[0\]: unknown field"):
            from_dict(
                AgentTestCase, {"id": "c", "input": [{"role": "user", "content": "x", "z": 1}]}
            )


class TestNoCodeExecution:
    def test_import_path_looking_strings_import_nothing(self, tmp_path, monkeypatch):
        marker = tmp_path / "imported"
        (tmp_path / "evil_module.py").write_text(
            f"open({str(marker)!r}, 'w').write('x')\ndef fn(**kw):\n    return 1\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        data = {
            "id": "c",
            "input": "evil_module:fn",
            "extra": {"handler": "evil_module:fn"},
            "fakes": [
                {
                    "name": "evil_module:fn",
                    "description": "evil_module.fn",
                    "extra": {"handler": "evil_module:fn"},
                }
            ],
            "expect": {"answer_contains": ["__import__('evil_module')"]},
        }
        case = from_dict(AgentTestCase, data)
        assert from_dict(AgentTestCase, to_dict(case)) == case
        assert not marker.exists() and "evil_module" not in sys.modules

    def test_a_handler_field_is_an_unknown_field(self):
        data = {"id": "c", "input": "hi", "fakes": [{"name": "x", "handler": "pkg.mod:fn"}]}
        with pytest.raises(
            ModelValidationError, match=r"fakes\[0\]: unknown field\(s\) \['handler'\]"
        ):
            from_dict(AgentTestCase, data)


def test_run_result_documents_round_trip_through_json_text():
    result = rich_result()
    text = json.dumps(to_dict(result), ensure_ascii=False, allow_nan=False)
    assert from_dict(RunResult, json.loads(text)) == result


class TestPhase1BShorthands:
    def expect(self, **fields):
        return from_dict(
            AgentTestCase, {"id": "c", "input": "hi", "use_judge": ["j"], "expect": fields}
        ).expect

    def test_routes_accepts_a_list_of_names_or_an_object(self):
        from disseqt_agent_testing.models import RouteExpect

        assert self.expect(routes=["billing", "refunds"]).routes == RouteExpect(
            path=("billing", "refunds")
        )
        assert self.expect(routes={"path": ["a"], "mode": "first"}).routes == RouteExpect(
            path=("a",), mode="first"
        )
        with pytest.raises(ModelValidationError, match="expect.routes"):
            self.expect(routes="billing")

    def test_grounded_accepts_true_or_an_object(self):
        from disseqt_agent_testing.models import GroundednessExpect

        assert self.expect(grounded=True).grounded == GroundednessExpect()
        assert self.expect(grounded={"allow_unsupported": True}).grounded.allow_unsupported  # type: ignore[union-attr]
        assert self.expect(grounded=False).grounded is None

    def test_intent_accepts_one_string_or_a_list(self):
        assert self.expect(intent="billing", intent_labels=["billing"]).intents == ("billing",)
        assert self.expect(intent=["a", "b"], intent_source="route").intents == ("a", "b")

    def test_matcher_lists_accept_plain_values(self):
        data = {
            "name": "x",
            "args": {"a": {"$m": "unordered", "items": ["a", {"$m": "regex", "pattern": "^b"}, 3]}},
        }
        tool = from_dict(ToolExpect, data)
        assert [type(m).__name__ for m in tool.args["a"].items] == ["Eq", "Regex", "Eq"]
        assert to_dict(tool)["args"]["a"]["items"] == [
            "a",
            {"$m": "regex", "pattern": "^b", "flags": 0},
            3,
        ]

    def test_dict_match_nests_matchers(self):
        data = {
            "name": "x",
            "args": {
                "a": {
                    "$m": "dict_match",
                    "fields": {"city": "NYC", "n": {"$m": "type_is", "types": ["integer"]}},
                }
            },
        }
        match = from_dict(ToolExpect, data).args["a"]
        assert match.fields["city"] == Eq("NYC") and match.mode == "subset"

    def test_semantic_matcher_needs_use_judge_on_load(self):
        data = {
            "id": "c",
            "input": "hi",
            "expect": {
                "tools": [{"name": "x", "args": {"a": {"$m": "semantic", "reference": "n"}}}]
            },
        }
        with pytest.raises(ModelValidationError, match="AgentTestCase.use_judge: must be set"):
            from_dict(AgentTestCase, data)
        assert from_dict(AgentTestCase, data | {"use_judge": ["semantic_arg"]})

    def test_old_files_without_1b_fields_still_load(self):
        old = to_dict(AgentTestCase("c", "hi", expect=Expect(max_steps=3)))
        for key in ("max_llm_calls", "routes", "loops", "intent", "intent_labels", "intent_source",
                    "clarify", "refuse", "scope", "plan", "answer_quality", "grounded"):  # fmt: skip
            old["expect"].pop(key)
        assert from_dict(AgentTestCase, old).expect.max_steps == 3

    def test_tool_expect_without_1b_fields_still_loads(self):
        assert from_dict(ToolExpect, {"name": "x"}).weight == 1.0

    def test_bad_blocks_report_a_path(self):
        with pytest.raises(ModelValidationError, match=r"LoopExpect.max_period: must be >= 2"):
            self.expect(loops={"max_period": 1})
        with pytest.raises(
            ModelValidationError, match=r"expect.clarify: unknown field\(s\) \['ask'\]"
        ):
            self.expect(clarify={"ask": True})


class TestHostileFiles:
    def nested_matchers(self, depth):
        root: dict = {"$m": "unordered", "items": []}
        node = root
        for _ in range(depth):
            child: dict = {"$m": "unordered", "items": []}
            node["items"].append(child)
            node = child
        return root

    def test_very_deep_nesting_is_a_validation_error_not_a_recursion_error(self):
        data = {"name": "x", "args": {"a": self.nested_matchers(3000)}}
        with pytest.raises(ModelValidationError, match="nesting is deeper than 64 levels"):
            from_dict(ToolExpect, data)

    def test_reasonable_nesting_still_loads(self):
        data = {"name": "x", "args": {"a": self.nested_matchers(15)}}
        assert from_dict(ToolExpect, data).args["a"].items

    def test_deep_json_values_are_bounded_too(self):
        deep: list = []
        node = deep
        for _ in range(3000):
            child: list = []
            node.append(child)
            node = child
        with pytest.raises(ModelValidationError, match="deeper than"):
            from_dict(ToolExpect, {"name": "x", "extra": {"a": deep}})

    def test_regex_flags_in_a_file_are_a_validation_error(self):
        data = {"name": "x", "args": {"a": {"$m": "regex", "pattern": "a", "flags": 4}}}
        with pytest.raises(ModelValidationError, match="Regex.flags"):
            from_dict(ToolExpect, data)

    def test_depth_counter_is_reset_between_calls(self):
        data = {"name": "x", "args": {"a": self.nested_matchers(15)}}
        for _ in range(5):
            from_dict(ToolExpect, data)


class TestDecoderRobustness:
    def test_from_dict_needs_a_model_class(self):
        with pytest.raises(TypeError, match="from_dict expects a model class"):
            from_dict(dict, {})  # type: ignore[type-var]
        with pytest.raises(TypeError, match="from_dict expects a model class"):
            from_dict(AgentTestCase("c", "hi"), {})  # type: ignore[type-var]

    def test_a_failed_union_attempt_does_not_shift_filled_call_ids(self):
        from dataclasses import dataclass

        @dataclass(slots=True, frozen=True)
        class Probe:
            items: "tuple[MsgToolCall, ...] | tuple[MsgToolCall, int]"

        # The first shape decodes one call and then fails on the integer; the second shape
        # re-decodes that call. Its filled id must still be call_1, not call_2.
        probe = from_dict(Probe, {"items": [{"name": "a"}, 5]}, fill_ids=True)
        assert probe.items[0].id == "call_1" and probe.items[1] == 5
