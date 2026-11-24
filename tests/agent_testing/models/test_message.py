"""Message, MsgToolCall, from_wire and validate_tool_correlation."""

import dataclasses

import pytest

from disseqt_agent_testing.models._frozen import FrozenDict
from disseqt_agent_testing.models.errors import ModelValidationError
from disseqt_agent_testing.models.message import Message, MsgToolCall, validate_tool_correlation


def call(call_id="c1", name="get_order", **arguments):
    return MsgToolCall(call_id, name, arguments=arguments)


class TestMessage:
    def test_defaults(self):
        message = Message("assistant")
        assert message.content == "" and message.tool_calls == ()
        assert message.tool_call_id is None and message.name is None

    def test_none_content_is_coerced(self):
        assert Message("assistant", None).content == ""  # type: ignore[arg-type]

    def test_positional_role_and_content_only(self):
        assert Message("user", "hi").content == "hi"
        with pytest.raises(TypeError):
            Message("user", "hi", ())  # type: ignore[misc]

    @pytest.mark.parametrize("role", ["user", "system"])
    def test_user_and_system_need_content(self, role):
        with pytest.raises(ModelValidationError, match="Message.content: must not be empty"):
            Message(role, "  ")

    def test_assistant_and_tool_may_be_empty(self):
        Message("assistant", "")
        Message("tool", "", tool_call_id="c1")

    def test_unknown_role(self):
        with pytest.raises(ModelValidationError, match="Message.role"):
            Message("function", "x")  # type: ignore[arg-type]

    @pytest.mark.parametrize("role", ["user", "system", "tool"])
    def test_tool_calls_only_on_assistant(self, role):
        kwargs = {"tool_call_id": "c1"} if role == "tool" else {}
        with pytest.raises(ModelValidationError, match="only allowed on assistant"):
            Message(role, "x", tool_calls=(call(),), **kwargs)

    def test_tool_message_needs_call_id(self):
        with pytest.raises(ModelValidationError, match="Message.tool_call_id"):
            Message("tool", "result")
        with pytest.raises(ModelValidationError, match="Message.tool_call_id"):
            Message("tool", "result", tool_call_id="")

    @pytest.mark.parametrize("role", ["user", "system", "assistant"])
    def test_call_id_only_on_tool_messages(self, role):
        with pytest.raises(ModelValidationError, match="only allowed on tool messages"):
            Message(role, "x", tool_call_id="c1")

    def test_duplicate_call_ids_rejected(self):
        with pytest.raises(ModelValidationError, match=r"repeated: \['c1'\]"):
            Message("assistant", tool_calls=(call("c1"), call("c1", "other")))

    def test_lists_become_tuples_and_items_are_checked(self):
        message = Message("assistant", tool_calls=[call()])  # type: ignore[arg-type]
        assert isinstance(message.tool_calls, tuple)
        with pytest.raises(ModelValidationError, match=r"tool_calls\[0\]"):
            Message("assistant", tool_calls=("get_order",))  # type: ignore[arg-type]

    def test_immutable_hashable_and_ordered(self):
        message = Message("assistant", tool_calls=(call("a", "x", q=1), call("b", "y")))
        assert [c.id for c in message.tool_calls] == ["a", "b"]
        hash(message)
        with pytest.raises(dataclasses.FrozenInstanceError):
            message.content = "x"  # type: ignore[misc]
        with pytest.raises(TypeError):
            message.tool_calls[0].arguments["q"] = 2

    def test_equal_content_equal_hash(self):
        a = Message("assistant", tool_calls=(call("a", "x", q=1),))
        b = Message("assistant", tool_calls=(call("a", "x", q=1),))
        assert a == b and hash(a) == hash(b)


class TestMsgToolCall:
    def test_required_fields(self):
        for bad_id, bad_name in (("", "x"), ("c", ""), (None, "x"), ("c", None)):
            with pytest.raises(ModelValidationError, match="MsgToolCall"):
                MsgToolCall(bad_id, bad_name)  # type: ignore[arg-type]

    def test_arguments_are_frozen_json(self):
        c = MsgToolCall("c", "x", arguments={"a": [1, {"b": 2}]})
        assert isinstance(c.arguments, FrozenDict) and c.arguments["a"][0] == 1

    @pytest.mark.parametrize(
        "bad", [{"a": float("nan")}, {1: "x"}, {"a": object()}, {"a": {"b": {1, 2}}}]
    )
    def test_arguments_must_be_json(self, bad):
        with pytest.raises(ModelValidationError, match="MsgToolCall.arguments"):
            MsgToolCall("c", "x", arguments=bad)

    def test_arguments_must_be_an_object(self):
        with pytest.raises(ModelValidationError, match="expected an object"):
            MsgToolCall("c", "x", arguments=[1])  # type: ignore[arg-type]

    def test_malformed_record_rules(self):
        with pytest.raises(ModelValidationError, match="raw_arguments: must be set"):
            MsgToolCall("c", "x", parse_error="boom")
        with pytest.raises(ModelValidationError, match="only allowed when parse_error"):
            MsgToolCall("c", "x", raw_arguments="{")
        with pytest.raises(ModelValidationError, match="arguments: must be empty"):
            MsgToolCall("c", "x", arguments={"a": 1}, raw_arguments="{", parse_error="boom")


class TestFromWire:
    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_empty_becomes_empty_object(self, empty):
        built = MsgToolCall.from_wire("c", "x", empty)
        assert built.arguments == {} and built.parse_error is None

    def test_dict_passes_through(self):
        assert MsgToolCall.from_wire("c", "x", {"a": 1}).arguments == {"a": 1}

    def test_string_is_parsed_once(self):
        built = MsgToolCall.from_wire("c", "x", '{"b": 1, "a": [1, 2]}')
        assert built.arguments == {"b": 1, "a": (1, 2)}
        assert list(built.arguments) == ["b", "a"]

    @pytest.mark.parametrize("text", ['{"a": ', "not json", '{"a": NaN}', '{"a": Infinity}'])
    def test_invalid_json_is_recorded_not_raised(self, text):
        built = MsgToolCall.from_wire("c", "x", text)
        assert built.arguments == {} and built.raw_arguments == text
        assert built.parse_error

    @pytest.mark.parametrize(("text", "kind"), [("[1]", "list"), ("3", "int"), ('"s"', "str")])
    def test_non_object_json(self, text, kind):
        built = MsgToolCall.from_wire("c", "x", text)
        assert built.parse_error == f"expected a JSON object, got {kind}"
        assert built.raw_arguments == text

    def test_non_json_dict_is_recorded_not_raised(self):
        built = MsgToolCall.from_wire("c", "x", {"a": object()})
        assert built.arguments == {} and built.parse_error and built.raw_arguments

    def test_deeply_nested_text_does_not_raise(self):
        built = MsgToolCall.from_wire("c", "x", "[" * 5000)
        assert built.parse_error

    def test_id_and_name_are_still_required(self):
        with pytest.raises(ModelValidationError, match="MsgToolCall.id"):
            MsgToolCall.from_wire("", "x", "{}")


class TestCorrelation:
    def exchange(self, *answers):
        calls = Message("assistant", tool_calls=(call("a", "x"), call("b", "y")))
        return [Message("user", "go"), calls, *answers]

    def test_accepts_answers_in_any_order(self):
        for order in (("a", "b"), ("b", "a")):
            history = self.exchange(*(Message("tool", "r", tool_call_id=i) for i in order))
            validate_tool_correlation([*history, Message("assistant", "done")])

    def test_accepts_history_without_tools(self):
        validate_tool_correlation([Message("user", "hi"), Message("assistant", "hello")])

    def test_missing_answer(self):
        history = self.exchange(Message("tool", "r", tool_call_id="a"))
        with pytest.raises(ModelValidationError, match=r"\['b'\] were not answered"):
            validate_tool_correlation([*history, Message("assistant", "done")])

    def test_missing_answer_at_end_of_history(self):
        with pytest.raises(ModelValidationError, match="were not answered"):
            validate_tool_correlation(self.exchange())

    def test_unknown_call_id(self):
        history = self.exchange(Message("tool", "r", tool_call_id="zzz"))
        with pytest.raises(ModelValidationError, match="'zzz', which is not a call"):
            validate_tool_correlation(history)

    def test_non_tool_message_between_call_and_answers(self):
        history = [
            *self.exchange(Message("tool", "r", tool_call_id="a")),
            Message("user", "interrupt"),
            Message("tool", "r", tool_call_id="b"),
        ]
        with pytest.raises(ModelValidationError, match="were not answered"):
            validate_tool_correlation(history)

    def test_answered_twice(self):
        history = self.exchange(
            Message("tool", "r", tool_call_id="a"), Message("tool", "r", tool_call_id="a")
        )
        with pytest.raises(ModelValidationError, match="answered more than once"):
            validate_tool_correlation(history)

    def test_orphan_tool_message(self):
        with pytest.raises(ModelValidationError, match="not a call of the preceding"):
            validate_tool_correlation(
                [Message("user", "x"), Message("tool", "r", tool_call_id="a")]
            )
