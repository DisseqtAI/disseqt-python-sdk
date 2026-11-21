"""Case-file loading (LD-): lines, errors, duplicates, datasets, filters, formats."""

import pytest

pytest.importorskip("yaml")  # YAML case files need the [yaml] extra
import json
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.runner import (
    CaseFilter,
    _yaml,
    format_errors,
    load_cases,
    register_case_format,
    register_dataset_format,
)
from disseqt_agent_testing.runner._json import JsonProblem, parse_json
from disseqt_agent_testing.runner.loader import (
    unregister_case_format,
    unregister_dataset_format,
)

BASE = """\
version: 1
defaults:
  repeats: 5
  tags: [refund]
  run: {agent: "tests.agent_testing._support.runner_agents:make_good_agent", timeout_s: 30}
cases:
  - id: refund_happy
    input: "Refund order A12"
    fakes:
      get_order:
        effect: read
        responses:
          - {when: {order_id: A12}, returns: {order_id: A12, country: no}}
      issue_refund: {default: {returns: {amount: 59.99}}}
    expect:
      decision: call
      tools:
        - {name: get_order, id: lookup}
        - name: issue_refund
          args: {order_id: {"$m": "ref", source: "call:lookup.result.order_id"}}
          after: [lookup]
      order: partial
  - id: refund_missing
    input: "I want a refund"
    tags: [clarification, smoke]
    follow_ups: ["It's A12"]
    repeats: 3
    pass_k: 3
    pass_k_threshold: 0.8
    expect: {decision: ask, decision_by_turn: {1: call}, allowed_tools: ["*"]}
"""


def write(tmp: Path, name: str, text: str) -> Path:
    path = tmp / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def msgs(loaded: Any) -> list[str]:
    return [e.message for e in loaded.errors]


def test_ld01_a_valid_file_loads_with_lines_and_merged_defaults(tmp_path: Path) -> None:
    path = write(tmp_path, "refund.agent.yaml", BASE)
    loaded = load_cases(path)
    assert not loaded.errors and [c.case.id for c in loaded.cases] == [
        "refund_happy",
        "refund_missing",
    ]
    happy, missing = loaded.cases
    assert happy.source.line == 7 and missing.source.line == 23 and happy.source.index == 0
    assert happy.source.nodeid == f"{path}::refund_happy"
    assert happy.case.repeats == 5 and missing.case.repeats == 3  # the case wins over defaults
    assert happy.case.tags == ("refund",) and set(missing.case.tags) == {
        "refund",
        "clarification",
        "smoke",
    }
    assert (
        happy.run.agent == "tests.agent_testing._support.runner_agents:make_good_agent"
        and happy.run.timeout_s == 30
    )
    assert missing.case.follow_ups == ("It's A12",) and missing.case.pass_k_threshold == 0.8


def test_ld02_a_decoder_error_names_the_file_line_and_path(tmp_path: Path) -> None:
    text = BASE.replace("        - {name: get_order, id: lookup}\n", "        - {id: lookup}\n")
    loaded = load_cases(write(tmp_path, "bad.agent.yaml", text))
    assert len(loaded.errors) == 1 and "refund_missing" in [c.case.id for c in loaded.cases]
    assert msgs(loaded)[0].endswith("cases[0].expect.tools[0].name: required field is missing")
    assert ":18:" in msgs(loaded)[0] or ":19:" in msgs(loaded)[0]


def test_ld03_a_yaml_syntax_error_reports_its_line(tmp_path: Path) -> None:
    loaded = load_cases(
        write(tmp_path, "x.agent.yaml", "version: 1\ncases:\n  - id: a\n   input: [unclosed\n")
    )
    assert (
        not loaded.cases
        and "invalid YAML:" in msgs(loaded)[0]
        and ".agent.yaml:" in msgs(loaded)[0]
    )


def test_ld04_ld05_duplicates_are_errors_naming_the_first_definition(tmp_path: Path) -> None:
    one = """version: 1
cases:
  - {id: a, input: x}
  - {id: a, input: y}
"""
    loaded = load_cases(write(tmp_path, "d.agent.yaml", one))
    assert [c.case.id for c in loaded.cases] == ["a"]  # the first stays collectable
    assert msgs(loaded)[0].endswith("duplicate case id 'a' (first defined at 3)")
    p1 = write(
        tmp_path, "one.agent.json", json.dumps({"version": 1, "cases": [{"id": "z", "input": "q"}]})
    )
    p2 = write(
        tmp_path, "two.agent.json", json.dumps({"version": 1, "cases": [{"id": "z", "input": "q"}]})
    )
    both = load_cases([p1, p2])
    assert len(both.cases) == 1 and f"(first defined at {p1}:1)" in msgs(both)[0]


def test_ld06_yaml_booleans_follow_yaml_1_2(tmp_path: Path) -> None:
    loaded = load_cases(write(tmp_path, "refund.agent.yaml", BASE))
    returns = loaded.cases[0].case.fakes[0].responses[0].returns
    assert returns["country"] == "no" and returns["order_id"] == "A12"  # a string, not False
    text = "version: 1\ncases:\n  - id: t\n    input: x\n    tags: [a]\n    extra: {on: yes, flag: true, off: false}\n"
    case = load_cases(write(tmp_path, "b.agent.yaml", text)).cases[0].case
    assert dict(case.extra) == {"on": "yes", "flag": True, "off": False}


def test_ld07_json_duplicate_keys_are_rejected_with_a_line(tmp_path: Path) -> None:
    text = '{\n  "version": 1,\n  "cases": [],\n  "cases": []\n}\n'
    loaded = load_cases(write(tmp_path, "d.agent.json", text))
    assert (
        not loaded.cases
        and "invalid JSON: duplicate key 'cases'" in msgs(loaded)[0]
        and ":4:" in msgs(loaded)[0]
    )
    with pytest.raises(JsonProblem):
        parse_json('{"a": 1, "a": 2}')
    value, lines = parse_json('{"x": [1, {"y": 2}]}')
    assert value == {"x": [1, {"y": 2}]} and lines[("x", 1, "y")] == 1


def test_json_syntax_errors_carry_lines() -> None:
    for bad, fragment in [
        ('{"a": }', "Expecting value"),
        ('{"a" 1}', "':'"),
        ("[1 2]", "','"),
        ("{} extra", "Extra data"),
        ("", "Expecting value"),
    ]:
        with pytest.raises(JsonProblem) as err:
            parse_json(bad)
        assert fragment in err.value.problem and err.value.line == 1


def test_ld08_invalid_ids_and_tags(tmp_path: Path) -> None:
    text = """version: 1
cases:
  - {id: "bad id", input: x}
  - {id: ok, input: x, tags: [skip, "has space", good]}
"""
    out = msgs(load_cases(write(tmp_path, "t.agent.yaml", text)))
    assert any(
        "case id 'bad id' is invalid; use letters, digits, '_', '.', '-' (max 128)" in m
        for m in out
    )
    assert sum("is invalid or reserved" in m for m in out) == 2 and any(
        "tag 'skip'" in m for m in out
    )
    long_id = "a" * 129
    assert any(
        "is invalid" in m
        for m in msgs(
            load_cases(
                write(
                    tmp_path,
                    "l.agent.yaml",
                    f"version: 1\ncases:\n  - {{id: {long_id}, input: x}}\n",
                )
            )
        )
    )


def test_ld09_ld10_ld11_pass_k_unknown_keys_and_versions(tmp_path: Path) -> None:
    pk = "version: 1\ncases:\n  - {id: a, input: x, repeats: 2, pass_k: 3}\n"
    assert (
        "case 'a': pass_k=3 exceeds repeats=2"
        in msgs(load_cases(write(tmp_path, "p.agent.yaml", pk)))[0]
    )
    th = "version: 1\ncases:\n  - {id: a, input: x, pass_k_threshold: 1.5}\n"
    assert (
        "pass_k_threshold=1.5 must be in (0, 1]"
        in msgs(load_cases(write(tmp_path, "t.agent.yaml", th)))[0]
    )
    unk = "version: 1\nbogus: 1\ncases: []\n"
    assert (
        "unknown top-level key 'bogus' (allowed: version, defaults, cases)"
        in msgs(load_cases(write(tmp_path, "u.agent.yaml", unk)))[0]
    )
    ver = load_cases(write(tmp_path, "v.agent.yaml", "version: 2\ncases: []\n"))
    assert msgs(ver) == [
        f"{tmp_path / 'v.agent.yaml'}:1: case file version 2 is not supported (supported: 1)"
    ]
    assert (
        "version None" in msgs(load_cases(write(tmp_path, "w.agent.yaml", "cases: []\n")))[0]
        or "None" in msgs(load_cases(write(tmp_path, "w2.agent.yaml", "cases: []\n")))[0]
    )


DATASET = """version: 1
cases:
  - id: "refund_${name}"
    input: "Refund ${name}"
    tags: [data]
    extra: {amount: "${amount}", label: "pay $${name}"}
    rows:
      - {name: A1, amount: 10}
      - {name: B2, amount: 20.5}
"""


def test_ld12_inline_rows_expand_to_cases_with_typed_values(tmp_path: Path) -> None:
    loaded = load_cases(write(tmp_path, "d.agent.yaml", DATASET))
    assert not loaded.errors and [c.case.id for c in loaded.cases] == ["refund_A1", "refund_B2"]
    first, second = loaded.cases
    assert first.case.input[-1].content == "Refund A1"
    assert dict(first.case.extra) == {
        "amount": 10,
        "label": "pay ${name}",
    }  # whole value keeps its type; $${ escapes
    assert second.case.extra["amount"] == 20.5
    assert (first.source.index, second.source.index) == (0, 1)


def test_ld13_csv_and_jsonl_datasets(tmp_path: Path) -> None:
    (tmp_path / "rows.csv").write_text("name,amount\nA1,10\nB2,20\n")
    (tmp_path / "rows.jsonl").write_text(
        '{"name": "C3", "amount": 5}\n\n{"name": "D4", "amount": 6}\n'
    )
    head = 'version: 1\ncases:\n  - id: "t_${name}"\n    input: "x ${name}"\n    extra: {amount: "${amount}"}\n    dataset: '
    csv_cases = load_cases(write(tmp_path, "c.agent.yaml", head + "rows.csv\n")).cases
    assert [c.case.id for c in csv_cases] == ["t_A1", "t_B2"] and csv_cases[0].case.extra[
        "amount"
    ] == "10"  # CSV values are strings
    jsonl = load_cases(write(tmp_path, "j.agent.yaml", head + "rows.jsonl\n")).cases
    assert [c.case.id for c in jsonl] == ["t_C3", "t_D4"] and jsonl[1].case.extra["amount"] == 6


def test_ld14_dataset_errors_name_the_row_and_the_cap(tmp_path: Path) -> None:
    unknown = DATASET.replace("amount: 20.5", "cost: 1")
    out = msgs(load_cases(write(tmp_path, "u.agent.yaml", unknown)))
    assert any("rows row 1: unknown field ${amount} in case.extra.amount" in m for m in out)
    noid = DATASET.replace('id: "refund_${name}"', "id: refund_x")
    assert any(
        "template id 'refund_x' has no ${...} placeholder" in m
        for m in msgs(load_cases(write(tmp_path, "n.agent.yaml", noid)))
    )
    rows = "\n".join(f"      - {{name: n{i}}}" for i in range(10_001))
    big = f'version: 1\ncases:\n  - id: "c_${{name}}"\n    input: x\n    rows:\n{rows}\n'
    assert any(
        "rows has 10001 rows (max 10000)" in m
        for m in msgs(load_cases(write(tmp_path, "b.agent.yaml", big)))
    )
    missing = 'version: 1\ncases:\n  - id: "c_${name}"\n    input: x\n    dataset: nope.csv\n'
    assert any(
        "dataset nope.csv" in m for m in msgs(load_cases(write(tmp_path, "m.agent.yaml", missing)))
    )


def test_ld15_twenty_five_errors_print_twenty_and_a_summary(tmp_path: Path) -> None:
    cases = "\n".join(f'  - {{id: "bad id {i}", input: x}}' for i in range(25))
    loaded = load_cases(write(tmp_path, "many.agent.yaml", f"version: 1\ncases:\n{cases}\n"))
    assert len(loaded.errors) == 25
    shown = format_errors(loaded.errors)
    assert len(shown) == 21 and shown[-1] == "... 5 more errors"


def test_ld16_ld17_filters(tmp_path: Path) -> None:
    path = write(tmp_path, "refund.agent.yaml", BASE)
    assert [c.case.id for c in load_cases(path, filter=CaseFilter(ids=("refund_h*",))).cases] == [
        "refund_happy"
    ]
    assert [
        c.case.id for c in load_cases(path, filter=CaseFilter(exclude_ids=("*happy",))).cases
    ] == ["refund_missing"]
    f = CaseFilter.build(tags=["smoke,nope", "!slow"])
    assert [c.case.id for c in load_cases(path, filter=f).cases] == ["refund_missing"]
    both = CaseFilter.build(tags=["refund", "smoke"])  # repeated entries are ANDed
    assert [c.case.id for c in load_cases(path, filter=both).cases] == ["refund_missing"]
    neg = load_cases(path, filter=CaseFilter.build(tags=["!clarification"]))
    assert [c.case.id for c in neg.cases] == ["refund_happy"] and neg.deselected == (
        "refund_missing",
    )
    assert CaseFilter().matches("anything") and not CaseFilter().active


def test_ld18_directories_are_walked_for_matching_files_only(tmp_path: Path) -> None:
    write(tmp_path, "a/one.agent.yaml", "version: 1\ncases:\n  - {id: one, input: x}\n")
    write(
        tmp_path,
        "a/b/two.agent.json",
        json.dumps({"version": 1, "cases": [{"id": "two", "input": "x"}]}),
    )
    write(tmp_path, "a/notes.yaml", "not: a case file\n")
    write(tmp_path, "a/data.json", "{}")
    loaded = load_cases(tmp_path / "a")
    assert (
        sorted(c.case.id for c in loaded.cases) == ["one", "two"]
        and not loaded.errors
        and len(loaded.files) == 2
    )
    assert load_cases(tmp_path / "missing").errors[0].message.endswith("no such file or directory")


def test_ld19_without_pyyaml_yaml_files_name_the_extra_while_json_still_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_yaml, "yaml", None)
    yml = write(tmp_path, "a.agent.yaml", "version: 1\ncases: []\n")
    js = write(
        tmp_path, "b.agent.json", json.dumps({"version": 1, "cases": [{"id": "ok", "input": "x"}]})
    )
    loaded = load_cases([yml, js])
    assert [c.case.id for c in loaded.cases] == ["ok"]
    assert (
        msgs(loaded)[0]
        == f"{yml}: YAML case files need the [yaml] extra: pip install 'disseqt-ai-sdk[yaml]'"
    )


def test_ld20_the_encoding_shims(tmp_path: Path) -> None:
    text = """version: 1
cases:
  - id: shim
    input: "a plain string"
    fakes:
      lookup: {effect: read, default: {returns: {ok: true}}}
      other: {kind: subagent, default: {returns: hi}}
"""
    case = load_cases(write(tmp_path, "s.agent.yaml", text)).cases[0].case
    assert case.input[-1].content == "a plain string" and [f.name for f in case.fakes] == [
        "lookup",
        "other",
    ]
    listed = """version: 1
cases:
  - id: listed
    input: x
    fakes:
      - {name: lookup, effect: read}
"""
    assert (
        load_cases(write(tmp_path, "l.agent.yaml", listed)).cases[0].case.fakes[0].name == "lookup"
    )


def test_ld21_the_run_block_is_strict(tmp_path: Path) -> None:
    ok = "version: 1\ncases:\n  - id: r\n    input: x\n    run: {timeout_s: 5, skip: 'later', requires: [yaml], xfail: why}\n"
    spec = load_cases(write(tmp_path, "r.agent.yaml", ok)).cases[0]
    assert (spec.run.timeout_s, spec.run.skip, spec.run.requires, spec.run.xfail) == (
        5,
        "later",
        ("yaml",),
        "why",
    )
    typo = "version: 1\ncases:\n  - id: r\n    input: x\n    run: {agnet: m:f}\n"
    assert (
        "RunBlock: unknown field(s) ['agnet']"
        in msgs(load_cases(write(tmp_path, "t.agent.yaml", typo)))[0]
    )
    bad_agent = "version: 1\ncases:\n  - id: r\n    input: x\n    run: {agent: not-an-import}\n"
    assert (
        "agent 'not-an-import' is not an import path of the form 'module:attr'"
        in msgs(load_cases(write(tmp_path, "g.agent.yaml", bad_agent)))[0]
    )
    bad_unknown = "version: 1\ncases:\n  - id: r\n    input: x\n    nope: 1\n"
    assert (
        "unknown field(s) ['nope']"
        in msgs(load_cases(write(tmp_path, "k.agent.yaml", bad_unknown)))[0]
    )


def test_python_defined_cases_and_a_shared_registry() -> None:
    from .._support.runner_agents import refund_case

    loaded = load_cases([refund_case("p1"), refund_case("p2")])
    assert [c.case.id for c in loaded.cases] == ["p1", "p2"] and loaded.cases[1].source.index == 1
    dup = load_cases([refund_case("p1"), refund_case("p1")])
    assert len(dup.cases) == 1 and "duplicate case id 'p1'" in dup.errors[0].message
    assert "expected an AgentTestCase" in load_cases([42]).errors[0].message


def test_e_register_and_remove_formats(tmp_path: Path) -> None:
    register_case_format(
        ".toy", lambda text: ({"version": 1, "cases": [{"id": text.strip(), "input": "x"}]}, {})
    )
    register_dataset_format(".toydata", lambda p: [{"name": "z"}])
    try:
        assert [c.case.id for c in load_cases(write(tmp_path, "t.toy", "from_toy\n")).cases] == [
            "from_toy"
        ]
        text = 'version: 1\ncases:\n  - id: "d_${name}"\n    input: x\n    dataset: rows.toydata\n'
        (tmp_path / "rows.toydata").write_text("")
        assert [c.case.id for c in load_cases(write(tmp_path, "d.agent.yaml", text)).cases] == [
            "d_z"
        ]
        with pytest.raises(ValueError, match="already registered"):
            register_case_format(".toy", lambda t: ({}, {}))
    finally:
        unregister_case_format(".toy")
        unregister_dataset_format(".toydata")
    assert "unsupported extension" in msgs(load_cases(write(tmp_path, "t2.toy", "x")))[0]


def test_duplicate_keys_are_compared_as_constructed_values() -> None:
    assert _yaml.parse_yaml("1: a\n'1': b\n")[0] == {1: "a", "1": "b"}
    merged = "a: &x {k: 1}\nb: &y {j: 2}\nc:\n  <<: *x\n  <<: *y\n  k: 3\n"
    assert _yaml.parse_yaml(merged)[0]["c"] == {"k": 3, "j": 2}
    for text, shown in (("true: a\nTrue: b\n", "True"), ("a: 1\nb: 2\na: 3\n", "a")):
        with pytest.raises(JsonProblem, match=f"duplicate key '{shown}'") as info:
            _yaml.parse_yaml(text)
        assert info.value.line == text.count("\n")
    with pytest.raises(JsonProblem, match="a mapping key must be a scalar"):
        _yaml.parse_yaml("[a]: 1\n")


def test_numbers_follow_the_yaml_1_2_core_schema() -> None:
    text = (
        "a: 010\nb: 0o10\nc: 0x1F\nd: -7\ne: 1e3\nf: .5\ng: -.inf\n"
        "h: 1_000\ni: 1:30\nj: 0b11\nk: no\nl: '5'\n"
    )
    value, _ = _yaml.parse_yaml(text)
    assert value == {
        "a": 10, "b": 8, "c": 31, "d": -7, "e": 1000.0, "f": 0.5, "g": float("-inf"),
        "h": "1_000", "i": "1:30", "j": "0b11", "k": "no", "l": "5",
    }  # fmt: skip
    assert type(value["a"]) is int and type(value["e"]) is float
    with pytest.raises(JsonProblem, match="invalid integer"):
        _yaml.parse_yaml("a: !!int nope\n")


HEAD = "version: 1\ncases:\n  - id: a\n    input: x\n"


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("utf8.agent.yaml", HEAD.encode() + b'    tags: ["\xff"]\n', ":1: invalid case file: 'utf-8'"),
        ("utf8.agent.json", b'{"version": 1, "cases": [{"id": "\xff"}]}', ":1: invalid case file:"),
        ("date.agent.yaml", HEAD + "    follow_ups: [2024-13-45]\n", ":5: invalid YAML: invalid date"),
        ("alias.agent.yaml", HEAD + "    follow_ups: &a [*a]\n", ":5: invalid YAML: an alias refers"),
        ("deep.agent.yaml", "a: " + "[" * 3000 + "]" * 3000, "nested too deeply"),
        ("deep.agent.json", "[" * 100_000, "nested too deeply"),
    ],
)  # fmt: skip
def test_an_unreadable_document_is_a_collection_error(
    tmp_path: Path, name: str, content: Any, expected: str
) -> None:
    path = tmp_path / name
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    loaded = load_cases(path)
    assert not loaded.cases and len(loaded.errors) == 1 and loaded.errors[0].code == "DSQ-C01"
    assert expected in loaded.errors[0].message


@pytest.mark.parametrize(
    ("defaults", "extra", "expected"),
    [
        ("{tags: [x]}", "tags: null", ":6: cases[0].tags: expected a list, got null"),
        ("{tags: [x]}", "tags: 5", ":6: cases[0].tags: expected a list, got int"),
        ("{}", "tags: 5", ":6: cases[0].tags: expected a list, got int"),
        ("{tags: [x]}", "tags: [{a: 1}]", "is invalid or reserved"),
        ("{tags: [x]}", "tags: [1]", "tag '1' is invalid or reserved"),
        ("{tags: 5}", "repeats: 1", ":2: defaults.tags: expected a list, got int"),
        ("{tags: abc}", "repeats: 1", ":2: defaults.tags: expected a list, got str"),
        ("{run: {timeout_s: 3}}", "run: 5", "run: RunBlock: expected a mapping"),
        ("{run: {timeout_s: 3}}", "run: [1]", "run: RunBlock: expected a mapping"),
    ],
)
def test_wrongly_typed_tags_and_run_blocks_are_collection_errors(
    tmp_path: Path, defaults: str, extra: str, expected: str
) -> None:
    text = f"version: 1\ndefaults: {defaults}\ncases:\n  - id: a\n    input: x\n    {extra}\n"
    loaded = load_cases(write(tmp_path, "c.agent.yaml", text))
    assert len(loaded.errors) == 1 and expected in loaded.errors[0].message, msgs(loaded)


def test_defaults_still_merge_with_a_case_that_has_its_own_tags_and_run(tmp_path: Path) -> None:
    text = (
        "version: 1\ndefaults: {tags: [smoke, base], run: {timeout_s: 3}}\ncases:\n"
        "  - {id: a, input: x, tags: [smoke, extra], run: {skip: later}}\n"
        "  - {id: b, input: x, run: null}\n"
    )
    loaded = load_cases(write(tmp_path, "c.agent.yaml", text))
    a, b = loaded.cases
    assert not loaded.errors and a.case.tags == ("base", "extra", "smoke")
    assert (a.run.timeout_s, a.run.skip) == (3, "later")
    assert b.case.tags == ("base", "smoke") and b.run.timeout_s == 3


def test_a_dataset_the_reader_cannot_parse_is_a_collection_error(tmp_path: Path) -> None:
    (tmp_path / "big.csv").write_text("name\n" + "x" * 200_000 + "\n")
    (tmp_path / "deep.jsonl").write_text("[" * 100_000 + "\n")
    for ref, expected in (
        ("big.csv", "dataset big.csv: field larger than field limit"),
        ("deep.jsonl", "dataset deep.jsonl: a row is nested too deeply"),
    ):
        text = f'version: 1\ncases:\n  - id: "t_${{name}}"\n    input: x\n    dataset: {ref}\n'
        loaded = load_cases(write(tmp_path, "d.agent.yaml", text))
        assert not loaded.cases and loaded.errors[0].code == "DSQ-C10"
        assert expected in loaded.errors[0].message
