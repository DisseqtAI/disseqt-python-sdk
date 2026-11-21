"""FrozenDict, freeze_json, capture_json."""

import copy
import json
import pickle

import pytest

from disseqt_agent_testing.models._frozen import FrozenDict, capture_json, freeze_json, thaw
from disseqt_agent_testing.models.errors import ModelValidationError


class TestFrozenDict:
    @pytest.mark.parametrize(
        "mutate",
        [
            lambda d: d.__setitem__("x", 1),
            lambda d: d.__delitem__("a"),
            lambda d: d.clear(),
            lambda d: d.pop("a"),
            lambda d: d.popitem(),
            lambda d: d.setdefault("z", 1),
            lambda d: d.update({"x": 1}),
            lambda d: d.__ior__({"x": 1}),
        ],
    )
    def test_every_mutator_raises(self, mutate):
        d = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="FrozenDict is read-only"):
            mutate(d)
        assert d == {"a": 1}

    def test_hash_and_equality(self):
        a = FrozenDict({"a": (1, 2), "b": FrozenDict({"c": 3})})
        b = FrozenDict({"b": FrozenDict({"c": 3}), "a": (1, 2)})
        assert a == b and hash(a) == hash(b)
        assert {a: "ok"}[b] == "ok"

    def test_copy_and_pickle_round_trip(self):
        d = FrozenDict({"a": (1, FrozenDict({"b": 2}))})
        assert copy.deepcopy(d) == d
        assert isinstance(copy.deepcopy(d), FrozenDict)
        assert pickle.loads(pickle.dumps(d)) == d

    def test_json_dumps_works(self):
        assert json.dumps(FrozenDict({"a": 1})) == '{"a": 1}'


class TestFreezeJson:
    def test_accepts_nested_structures(self):
        frozen = freeze_json({"a": [1, {"b": [True, None, 2.5, "x"]}]})
        assert isinstance(frozen, FrozenDict)
        assert frozen["a"][0] == 1
        assert isinstance(frozen["a"], tuple)
        assert isinstance(frozen["a"][1], FrozenDict)
        assert frozen["a"][1]["b"] == (True, None, 2.5, "x")
        hash(frozen)  # fully hashable

    def test_keeps_scalar_types(self):
        assert freeze_json(True) is True
        assert freeze_json(1) == 1 and type(freeze_json(1)) is int
        assert type(freeze_json(1.0)) is float

    def test_preserves_key_order(self):
        assert list(freeze_json({"z": 1, "a": 2, "m": 3})) == ["z", "a", "m"]

    @pytest.mark.parametrize(
        ("bad", "fragment"),
        [
            (float("nan"), "float('nan') is not JSON-able"),
            (float("inf"), "float('inf') is not JSON-able"),
            (b"x", "bytes is not JSON-able"),
            ({1, 2}, "set is not JSON-able"),
            (object(), "object is not JSON-able"),
            ({1: "x"}, "key 1 is not a string"),
        ],
    )
    def test_rejects_non_json(self, bad, fragment):
        with pytest.raises(ModelValidationError, match="ToolExpect.args") as err:
            freeze_json({"amount": bad}, "ToolExpect.args")
        assert fragment in str(err.value)

    def test_error_path_points_at_value(self):
        with pytest.raises(ModelValidationError) as err:
            freeze_json({"items": [1, float("nan")]}, "ToolExpect.args")
        assert str(err.value).startswith("ToolExpect.args.items[1]: ")

    def test_rejects_datetime(self):
        import datetime

        with pytest.raises(ModelValidationError, match="datetime is not JSON-able"):
            freeze_json(datetime.datetime(2026, 1, 1), "x")

    def test_depth_limit(self):
        deep: object = "x"
        for _ in range(150):
            deep = [deep]
        with pytest.raises(ModelValidationError, match="deeper than"):
            freeze_json(deep)

    def test_thaw_gives_mutable_copies(self):
        frozen = freeze_json({"a": [1, {"b": 2}]})
        plain = thaw(frozen)
        assert plain == {"a": [1, {"b": 2}]}
        plain["a"].append(3)
        assert frozen["a"] == (1, FrozenDict({"b": 2}))


class TestCaptureJson:
    def test_json_able_values_are_kept(self):
        value, lossy = capture_json({"a": [1, 2.5, None, "x", True]})
        assert value == {"a": (1, 2.5, None, "x", True)} and lossy is False

    def test_objects_become_repr_and_lossy(self):
        import datetime

        value, lossy = capture_json({"when": datetime.date(2026, 1, 2), "o": object()})
        assert lossy is True
        assert value["when"] == "datetime.date(2026, 1, 2)"
        assert value["o"].startswith("<object object at")

    def test_nan_and_infinity_become_strings(self):
        value, lossy = capture_json([float("nan"), float("inf"), float("-inf")])
        assert value == ("NaN", "Infinity", "-Infinity") and lossy is True

    def test_non_string_keys_are_stringified(self):
        value, lossy = capture_json({1: "a"})
        assert value == {"1": "a"} and lossy is True

    def test_cycle_is_cut(self):
        loop: list[object] = []
        loop.append(loop)
        value, lossy = capture_json(loop)
        assert lossy is True
        assert value is not None

    def test_repr_is_truncated(self):
        value, _ = capture_json(object.__new__(type("Big", (), {"__repr__": lambda s: "x" * 5000})))
        assert isinstance(value, str) and len(value) == 2000

    def test_hostile_repr_never_raises(self):
        class Bad:
            def __repr__(self) -> str:
                raise RuntimeError("boom")

        value, lossy = capture_json(Bad())
        assert value == "<unrepresentable Bad>" and lossy is True

    def test_sets_are_lossy(self):
        value, lossy = capture_json({1, 2})
        assert lossy is True and isinstance(value, str)

    def test_ints_over_the_size_limit_become_a_marker_and_lossy(self):
        value, lossy = capture_json({"n": 10**5000, "m": [-(10**5000)], "ok": 10**3999})
        assert lossy is True
        assert value["n"] == "<int with 16610 bits>" and value["m"] == ("<int with 16610 bits>",)
        assert value["ok"] == 10**3999
        assert freeze_json(value) == value  # what was captured can always be stored in a step
        assert capture_json(10**3999) == (10**3999, False)
        assert capture_json(True) == (True, False)


class TestFrozenDictHardening:
    def test_calling_init_again_cannot_mutate(self):
        frozen = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="FrozenDict is read-only"):
            frozen.__init__({"b": 2})
        assert frozen == {"a": 1}

    @pytest.mark.parametrize("bad", [[], {}, {1}, {"x": 1}])
    def test_unhashable_values_are_rejected_at_construction(self, bad):
        with pytest.raises(TypeError, match="values must be hashable"):
            FrozenDict({"a": bad})

    def test_hashable_values_are_accepted(self):
        value = FrozenDict({"a": (1, FrozenDict({"b": 2})), "c": "x", "d": None})
        hash(value)

    def test_empty_and_keyword_construction_still_work(self):
        assert FrozenDict() == {} and FrozenDict(a=1) == {"a": 1}


class TestSurrogates:
    SURROGATE = "bad \ud800 text"

    def test_freeze_json_rejects_lone_surrogates_in_values_and_keys(self):
        with pytest.raises(ModelValidationError, match="ToolExpect.args.a: string holds a lone"):
            freeze_json({"a": self.SURROGATE}, "ToolExpect.args")
        with pytest.raises(ModelValidationError, match="a key holds a lone surrogate"):
            freeze_json({self.SURROGATE: 1}, "x")

    def test_valid_non_ascii_text_is_fine(self):
        text = "Zoë 日本語 🚀"
        assert freeze_json({"k": text})["k"] == text
        assert json.dumps(freeze_json(text), ensure_ascii=False).encode("utf-8")

    def test_capture_json_replaces_them_and_marks_lossy(self):
        value, lossy = capture_json({self.SURROGATE: self.SURROGATE})
        assert lossy is True
        (key,) = value
        assert "\ud800" not in key and "\ud800" not in value[key]
        json.dumps(value, ensure_ascii=False).encode("utf-8")  # now writable

    def test_validated_strings_reject_them(self):
        from disseqt_agent_testing.models import AgentTestCase, Message

        with pytest.raises(ModelValidationError, match="lone surrogate"):
            Message("user", self.SURROGATE)
        with pytest.raises(ModelValidationError, match="lone surrogate"):
            AgentTestCase("c", "hi", follow_ups=(self.SURROGATE,))
