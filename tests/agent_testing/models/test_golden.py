"""Frozen golden files: saved data must keep loading, and keep its content when re-saved.

When a field is added later, do NOT edit these files; add a new ``*_plus_*.json`` fixture for the
new field (plans/01, section 3a.4). They are the guard that adding a parameter never breaks files
that already exist. Because every field is written explicitly, a newer version re-serializes an
old file with *extra* default fields; the guard is therefore "everything in the old file is still
there, unchanged and in the same order", not byte equality. (Message examples omit defaults, so
they stay byte-identical.)
"""

import json
from pathlib import Path

import pytest

from disseqt_agent_testing.models import AgentTestCase, Message, RunResult, from_dict, to_dict

from .factories import rich_case, rich_case_1b, rich_result

GOLDEN = Path(__file__).parent / "golden"


def dump(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def assert_preserved(old: object, new: object, path: str = "$") -> None:
    """``new`` contains all of ``old`` (same values, same relative key order); extras allowed."""
    if isinstance(old, dict):
        assert isinstance(new, dict), path
        missing = [k for k in old if k not in new]
        assert not missing, f"{path}: lost keys {missing}"
        assert [k for k in new if k in old] == list(old), f"{path}: key order changed"
        for key, value in old.items():
            assert_preserved(value, new[key], f"{path}.{key}")
    elif isinstance(old, list):
        assert isinstance(new, list) and len(old) == len(new), path
        for index, (a, b) in enumerate(zip(old, new, strict=True)):
            assert_preserved(a, b, f"{path}[{index}]")
    else:
        assert old == new and type(old) is type(new), f"{path}: {old!r} != {new!r}"


@pytest.mark.parametrize(
    ("name", "cls", "factory"),
    [
        ("case_v1.json", AgentTestCase, rich_case),
        ("case_v1_plus_1b.json", AgentTestCase, rich_case_1b),
        ("run_v1.json", RunResult, rich_result),
    ],
)
def test_golden_documents(name, cls, factory):
    text = (GOLDEN / name).read_text(encoding="utf-8")
    loaded = from_dict(cls, json.loads(text))
    assert loaded == factory()  # the saved file still loads to the same object
    assert_preserved(json.loads(text), to_dict(loaded))  # and keeps its content when re-saved
    assert from_dict(cls, to_dict(loaded)) == loaded


def test_golden_message_examples_match_the_documented_json():
    lines = (GOLDEN / "messages_v1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        message = from_dict(Message, json.loads(line))
        assert json.dumps(to_dict(message), ensure_ascii=False, allow_nan=False) == line
    first = from_dict(Message, json.loads(lines[0]))
    assert first.tool_calls[0].arguments == {"city": "SF"}
    third = from_dict(Message, json.loads(lines[2]))
    assert (
        third.tool_calls[0].parse_error == "Expecting value" and not third.tool_calls[0].arguments
    )


def test_golden_files_have_no_trailing_surprises():
    for path in GOLDEN.iterdir():
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n") and not text.endswith("\n\n"), path.name


def test_the_1b_golden_covers_every_phase_1b_field():
    data = json.loads((GOLDEN / "case_v1_plus_1b.json").read_text(encoding="utf-8"))
    expect = data["expect"]
    for key in (
        "max_llm_calls", "routes", "loops", "intent", "intent_labels", "intent_source", "clarify",
        "refuse", "scope", "plan", "answer_quality", "grounded",
    ):  # fmt: skip
        assert expect[key] not in (None, [], ""), key
    tools = expect["tools"]
    assert any(t["ignore"] for t in tools)
    assert any(t["same_step_as"] for t in tools)
    assert any(t["weight"] != 1.0 for t in tools)
    tags = {m["$m"] for m in _matchers(data) if isinstance(m, dict) and "$m" in m}
    assert {"normalized", "type_is", "unordered", "contains", "dict_match", "date_eq",
            "predicate", "semantic"} <= tags  # fmt: skip


def _matchers(value: object):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _matchers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _matchers(item)
