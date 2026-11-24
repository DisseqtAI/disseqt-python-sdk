"""Calibration: golden sets, split, statistics, gating thresholds, reports, CLI."""

import json
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.judges import KindConfig, make_judge
from disseqt_agent_testing.judges.calibration import (
    GoldenError,
    GoldenItem,
    corrected_rate,
    load_golden,
    load_reports,
    save_golden,
    split_of,
    validate_items,
    write_report,
)
from disseqt_agent_testing.judges.calibration.golden import (
    export_for_labeling,
    import_labels,
)
from disseqt_agent_testing.judges.calibration.metrics import (
    binary_stats,
    bootstrap_ci,
    kappa,
    multiclass_stats,
)
from disseqt_agent_testing.judges.calibration.run import calibrate, calibrate_kind
from disseqt_agent_testing.judges.cli import main as judges_cli
from disseqt_agent_testing.judges.models import FakeJudgeModel

from .helpers import clar

pytestmark = pytest.mark.filterwarnings("ignore::disseqt_agent_testing.judges.api.JudgeWarning")
KIND = "clarification_relevance"


def item(n: int, good: bool, *, source: str = "human") -> GoldenItem:
    return GoldenItem(
        f"i{n}", KIND,
        {"turn": 0, "conversation": [{"role": "user", "turn": 0, "text": "hi"}],
         "agent_text": f"{'GOOD' if good else 'BAD'} question {n}"},
        "pass" if good else "fail", source=source,
    )  # fmt: skip


def golden_set(n: int = 400) -> list[GoldenItem]:
    return [item(i, i % 2 == 0) for i in range(n)]


def fake(accuracy_flips: set[str] = frozenset()) -> FakeJudgeModel:  # type: ignore[assignment]
    def script(req: Any) -> Any:
        text = req.messages[-1]["content"]
        good = "GOOD" in text
        flip = any(f in text for f in accuracy_flips)
        return clar("pass" if good != flip else "fail")

    return FakeJudgeModel(script)


def engine_for(model: FakeJudgeModel, n: int = 1) -> Any:
    return make_judge(model, require_calibration="off", kinds={KIND: KindConfig(samples=n)},
                      backoff_base_s=0.0, cache_mode="off")  # fmt: skip


# ---------------------------------------------------------------- split and statistics
def test_split_is_stable_and_roughly_15_42_42() -> None:
    labels = [split_of(KIND, f"id{n}") for n in range(4000)]
    assert labels == [split_of(KIND, f"id{n}") for n in range(4000)]
    share = {s: labels.count(s) / len(labels) for s in ("dev", "validation", "test")}
    assert abs(share["dev"] - 0.15) < 0.03 and abs(share["test"] - 0.425) < 0.03
    assert split_of("a", "x") != split_of("b", "x") or split_of("a", "y") != split_of("b", "y")


def test_kappa_and_binary_stats() -> None:
    pairs = [("pass", "pass")] * 8 + [("fail", "fail")] * 8 + [("pass", "fail"), ("fail", "pass")]
    s = binary_stats(pairs, "pass")
    assert s["tpr"] == pytest.approx(8 / 9) and s["tnr"] == pytest.approx(8 / 9)
    assert kappa(pairs) == pytest.approx(0.7778, abs=1e-3)
    assert kappa([]) is None and kappa([("a", "a")] * 3) == 1.0


def test_multiclass_stats_and_macro_f1() -> None:
    pairs = [("a", "a")] * 5 + [("b", "b")] * 4 + [("b", "a")]
    s = multiclass_stats(pairs)
    assert s["per_class"]["b"]["recall"] == pytest.approx(0.8)
    assert s["macro_f1"] == pytest.approx((10 / 11 + 8 / 9) / 2, abs=1e-3)


def test_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    pairs = [("pass", "pass")] * 30 + [("pass", "fail")] * 5
    f = lambda p: binary_stats(p, "pass")["tpr"]  # noqa: E731
    a, b = bootstrap_ci(pairs, f), bootstrap_ci(pairs, f)
    assert a == b and a is not None and a[0] < 30 / 35 < a[1]
    assert bootstrap_ci([], f) is None


def test_prevalence_correction() -> None:
    assert corrected_rate(0.5, 1.0, 1.0) == 0.5
    assert corrected_rate(0.6, 0.9, 0.9) == pytest.approx(0.625)
    assert corrected_rate(0.5, 0.5, 0.5) is None  # a coin flip: no correction
    assert corrected_rate(0.99, 0.9, 0.9) == 1.0 and corrected_rate(0.0, 0.9, 0.9) == 0.0


# ---------------------------------------------------------------- golden files
def test_golden_roundtrip_and_validation(tmp_path: Path) -> None:
    items = golden_set(6)
    path = tmp_path / "g.jsonl"
    save_golden(path, items)
    assert load_golden(path) == items and validate_items(items, KIND) == []
    bad = [
        GoldenItem("x", KIND, {"turn": 0}, "pass"),
        GoldenItem("y", KIND, items[0].inputs, "nope"),
        GoldenItem("y", KIND, items[0].inputs, "unsure"),
        GoldenItem("z", "ghost_kind", {}, "pass"),
        GoldenItem("w", KIND, items[0].inputs, "pass", source="vibes"),
    ]
    text = "\n".join(validate_items(bad))
    for needle in (
        "inputs miss",
        "not one of",
        "duplicate id",
        "unknown judge kind",
        "source 'vibes'",
        "'unsure'",
    ):
        assert needle in text, needle
    path.write_text('{"schema_version": "9"}\n')
    with pytest.raises(GoldenError, match=":1:"):
        load_golden(path)


def test_labeling_workflow_requires_two_raters_and_adjudication(tmp_path: Path) -> None:
    items = golden_set(3)
    base = tmp_path / "out.csv"
    export_for_labeling(items, base)
    import csv

    with base.open(newline="") as fh:
        assert all(r["label"] == "" for r in csv.DictReader(fh))  # blind: no anchoring

    def rater(name: str, labels: list[str], adj: dict[int, str] | None = None) -> Path:
        path = tmp_path / f"{name}.csv"
        with base.open(newline="") as src, path.open("w", newline="") as dst:
            rows = list(csv.DictReader(src))
            w = csv.DictWriter(dst, rows[0].keys())
            w.writeheader()
            for n, row in enumerate(rows):
                row.update(rater=name, label=labels[n], adjudicated_label=(adj or {}).get(n, ""))
                w.writerow(row)
        return path

    a = rater("ann", ["pass", "fail", "pass"])
    b = rater("bob", ["pass", "pass", "pass"])
    result = import_labels([a, b])
    assert [i.id for i in result.items] == ["i0", "i2"] and len(result.needs_adjudication) == 1
    adjudicated = import_labels([a, rater("bob2", ["pass", "pass", "pass"], {1: "fail"})])
    one = next(i for i in adjudicated.items if i.id == "i1")
    assert one.label == "fail" and one.adjudicated and one.source == "human"
    assert import_labels([a]).items == () and len(import_labels([a]).skipped) == 3  # one rater


# ---------------------------------------------------------------- gating
def test_a_good_judge_on_enough_human_items_may_gate(tmp_path: Path) -> None:
    reports = calibrate(KIND, golden_set(), lambda n: engine_for(fake(), n), samples=(1,),
                        report_root=tmp_path)  # fmt: skip
    r = reports[0]
    assert r.gating_allowed and r.failures == () and r.stats["tpr"] == 1.0 and r.abstain_rate == 0.0
    assert r.counts["human_test"]["pass"] >= 30 and r.ci["tpr"] is not None
    path = write_report(r, tmp_path / "cal")
    assert path == tmp_path / "cal" / KIND / f"{r.identity_hash}.json"
    assert [x.identity_hash for x in load_reports(tmp_path / "cal")] == [r.identity_hash]


def test_gate_honours_each_threshold() -> None:
    items = golden_set()
    weak = calibrate_kind(engine_for(fake({"question 1"})), KIND, items)  # flips many verdicts
    assert not weak.gating_allowed and any("below" in f for f in weak.failures)
    few = calibrate_kind(engine_for(fake()), KIND, items[:40])
    assert not few.gating_allowed and any("human" in f for f in few.failures)
    drafts = [GoldenItem(i.id, i.kind, i.inputs, i.label, source="draft") for i in items]
    draft_report = calibrate_kind(engine_for(fake()), KIND, drafts)
    assert not draft_report.gating_allowed  # drafts are measured but never license gating
    assert draft_report.silver["tpr"] == 1.0 and draft_report.counts["human_test"] == {}


def test_abstentions_fail_the_gate_and_errors_count_as_abstentions() -> None:
    def script(req: Any) -> Any:
        return {
            "reasoning": "?",
            "label": "unsure",
            "items_requested": [],
            "asks_for_known_info": False,
        }

    r = calibrate_kind(engine_for(FakeJudgeModel(script)), KIND, golden_set(120))
    assert r.abstain_rate == 1.0 and not r.gating_allowed


def test_calibrating_n3_and_n1_gives_two_identities_and_the_test_split_ledger_warns(
    tmp_path: Path,
) -> None:
    reports = calibrate(KIND, golden_set(80), lambda n: engine_for(fake(), n), samples=(3, 1),
                        report_root=tmp_path)  # fmt: skip
    assert len({r.identity_hash for r in reports}) == 2 and [r.samples for r in reports] == [3, 1]
    assert any("already evaluated" in w for w in reports[1].warnings)
    again = calibrate_kind(engine_for(fake(), 3), KIND, golden_set(80), report_root=tmp_path)
    assert not any("already evaluated" in w for w in again.warnings)  # same identity: no warning


def test_a_report_written_by_calibration_unlocks_the_engine_gate(tmp_path: Path) -> None:
    from disseqt_agent_testing.assertions import JudgeTask

    r = calibrate_kind(engine_for(fake()), KIND, golden_set())
    write_report(r, tmp_path / "cal")
    task = JudgeTask(KIND, inputs=dict(item(1, True).inputs))
    gated = make_judge(fake(), calibration_dirs=(str(tmp_path / "cal"),),
                       kinds={KIND: KindConfig(samples=1)}, backoff_base_s=0.0, cache_mode="off")  # fmt: skip
    try:
        assert gated.judge(task).gating == "gate"
    finally:
        gated.close()
    other = make_judge(fake(), calibration_dirs=(str(tmp_path / "cal"),),
                       kinds={KIND: KindConfig(samples=2)}, backoff_base_s=0.0, cache_mode="off")  # fmt: skip
    try:
        assert other.judge(task).gating == "advisory"  # a different identity is not calibrated
    finally:
        other.close()


def test_root_cause_never_gates_even_with_perfect_agreement() -> None:
    items = [GoldenItem(f"r{n}", "root_cause",
                        {"failed_checks": ["x"], "conversation": [{"role": "user", "turn": 0, "text": "hi"}]},
                        "wrong_args", source="human") for n in range(200)]  # fmt: skip
    model = FakeJudgeModel(lambda req: {"reasoning": "r", "label": "wrong_args"})
    eng = make_judge(model, require_calibration="off", kinds={"root_cause": KindConfig(samples=1)},
                     backoff_base_s=0.0, cache_mode="off")  # fmt: skip
    r = calibrate_kind(eng, "root_cause", items)
    eng.close()
    assert not r.gating_allowed and r.failures == ("advisory kind: never gates",)
    assert r.stats["macro_f1"] == 1.0


def test_multiclass_needs_100_human_items() -> None:
    from disseqt_agent_testing.judges.calibration.report import gate_failures

    f = gate_failures("multiclass", {"macro_f1": 0.9, "kappa": 0.9, "per_class": {}},
                      abstain_rate=0.0, human_counts={"a": 50}, positive=None)  # fmt: skip
    assert f == ["only 50 human test items (need 100)"]
    ok = gate_failures("multiclass", {"macro_f1": 0.9, "kappa": 0.9, "per_class": {}},
                       abstain_rate=0.0, human_counts={"a": 100}, positive=None)  # fmt: skip
    assert ok == []
    weak = gate_failures("multiclass", {"macro_f1": 0.9, "kappa": 0.9,
                                        "per_class": {"a": {"recall": 0.5, "support": 20.0},
                                                      "b": {"recall": 0.1, "support": 3.0}}},
                         abstain_rate=0.0, human_counts={"a": 100}, positive=None)  # fmt: skip
    assert len(weak) == 1 and "'a'" in weak[0]  # small classes are exempt from the recall floor


# ---------------------------------------------------------------- CLI
def test_cli_golden_and_templates_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "g.jsonl"
    save_golden(path, golden_set(10))
    assert judges_cli(["golden", "validate", str(path), "--kind", KIND]) == 0
    assert "10 items" in capsys.readouterr().out
    bad = tmp_path / "b.jsonl"
    bad.write_text(json.dumps({**golden_set(1)[0].to_dict(), "label": "zzz"}) + "\n")
    assert judges_cli(["golden", "validate", str(bad)]) == 1
    csv_path = tmp_path / "label.csv"
    assert judges_cli(["golden", "export", "--golden", str(path), "--out", str(csv_path)]) == 0
    assert csv_path.exists()
    assert judges_cli(["templates", "check"]) == 0
    assert judges_cli(["report", "--dir", str(tmp_path / "none")]) == 0
    assert judges_cli(["nonsense"]) == 2


def test_cli_cache_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from disseqt_agent_testing.judges import JsonDirStore

    store = JsonDirStore(tmp_path / "c")
    store.put("fake", "m", "0f" * 32, {"key": "0f" * 32, "response": {"text": "x"}})
    assert judges_cli(["cache", "stats", "--dir", str(tmp_path / "c")]) == 0
    assert "1 entries" in capsys.readouterr().out
    assert judges_cli(["cache", "verify", "--dir", str(tmp_path / "c")]) == 0
    store.list()[0].write_text("not json")
    assert judges_cli(["cache", "verify", "--dir", str(tmp_path / "c")]) == 1
    assert (
        judges_cli(
            ["cache", "prune", "--dir", str(tmp_path / "c"), "--older-than-days", "0", "--dry-run"]
        )
        == 0
    )
    assert store.list()
    assert (
        judges_cli(["cache", "prune", "--dir", str(tmp_path / "c"), "--older-than-days", "0"]) == 0
    )
    assert not store.list()


def test_disseqt_command_mounts_the_judges_subcommand() -> None:
    from disseqt_agent_testing.cli import main

    assert main(["judges", "templates", "check"]) == 0


def test_each_binary_threshold_alone_fails_the_gate() -> None:
    from disseqt_agent_testing.judges.calibration.report import gate_failures

    good: dict[str, Any] = {"tpr": 0.9, "tnr": 0.9, "kappa": 0.7}
    counts = {"pass": 40, "fail": 40}

    def run(stats: dict[str, Any] | None = None, **kw: Any) -> list[str]:
        args: dict[str, Any] = {"abstain_rate": 0.1, "human_counts": counts, "positive": "pass"}
        args.update(kw)
        return gate_failures("binary", {**good, **(stats or {})}, **args)

    assert run() == []
    assert run({"tpr": 0.84}) == ["tpr 0.84 is below 0.85"]
    assert run({"tnr": 0.84}) == ["tnr 0.84 is below 0.85"]
    assert run({"kappa": 0.59}) == ["kappa 0.59 is below 0.6"]
    assert run({"kappa": None}) and run({"tpr": None})  # a missing statistic fails closed
    assert run(abstain_rate=0.16) == ["abstain rate 0.16 is above 0.15"]
    assert run(abstain_rate=None)
    assert run(human_counts={"pass": 29, "fail": 40}) == [
        "only 29 human positive test items (need 30)"
    ]
    assert run(human_counts={"pass": 40, "fail": 29}) == [
        "only 29 human negative test items (need 30)"
    ]
    assert run(needs_order_consistency=True, order_consistency=0.89)
    assert run(needs_order_consistency=True, order_consistency=0.9) == []
    assert run(needs_order_consistency=True, order_consistency=None)
    assert run(order_consistency=0.1) == []  # only kinds with two orderings need it


def test_shipped_drafts_are_valid_labelled_drafts_and_never_gate() -> None:
    from disseqt_agent_testing.judges.calibration import golden as g

    root = Path(g.__file__).parent / "drafts"
    files = sorted(root.glob("*.jsonl"))
    assert {f.stem for f in files} >= {"clarification_relevance", "semantic_arg", "decision",
                                       "refusal_alternative", "groundedness", "plan_quality",
                                       "answer_quality"}  # fmt: skip
    for f in files:
        items = load_golden(f)
        assert items and validate_items(items, f.stem) == [], f.name
        assert {i.source for i in items} == {"draft"}, f.name


def test_cli_calibrate_writes_a_report_and_exits_1_when_it_cannot_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from disseqt_agent_testing.judges.models.registry import register_provider, unregister_provider

    register_provider("clifake", lambda model_id, **o: fake(), override=True)
    try:
        golden = tmp_path / "g.jsonl"
        save_golden(golden, golden_set(20))  # far too few items to gate
        code = judges_cli(["calibrate", "--kind", KIND, "--golden", str(golden), "--model",
                           "clifake:m-1", "--samples", "1", "--out", str(tmp_path / "cal"),
                           "--cache-mode", "off"])  # fmt: skip
        out = capsys.readouterr().out
        assert code == 1 and "gating_allowed=False" in out and "human" in out
        assert len(load_reports(tmp_path / "cal")) == 1
        assert judges_cli(["calibrate", "--kind", KIND, "--golden", str(tmp_path / "missing.jsonl"),
                           "--model", "clifake:m-1"]) == 3  # fmt: skip
    finally:
        unregister_provider("clifake")


def test_macro_f1_counts_a_class_the_judge_never_gets_right() -> None:
    pairs = [("a", "a")] * 10 + [("b", "b")] * 10 + [("c", "a")] * 10 + [("a", "c")]
    s = multiclass_stats(pairs)
    per = s["per_class"]
    assert per["c"]["precision"] == 0.0 and per["c"]["recall"] == 0.0 and per["c"]["f1"] == 0.0
    assert s["macro_f1"] == pytest.approx((per["a"]["f1"] + per["b"]["f1"] + 0.0) / 3)
    assert s["macro_f1"] < 0.8  # below the multiclass gate floor
    perfect = multiclass_stats([("a", "a")] * 3 + [("b", "b")] * 3)
    assert perfect["macro_f1"] == 1.0


def test_a_golden_item_without_a_source_is_a_draft_and_never_gates(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    rows = []
    for i in golden_set():
        row = i.to_dict()
        del row["source"]
        rows.append(json.dumps(row))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    items = load_golden(path)
    assert {i.source for i in items} == {"draft"}
    assert GoldenItem("x", KIND, items[0].inputs, "pass").source == "draft"
    r = calibrate_kind(engine_for(fake()), KIND, items)
    assert not r.gating_allowed and r.counts["human_test"] == {}
    human = calibrate_kind(engine_for(fake()), KIND, golden_set())
    assert human.gating_allowed  # explicitly human-labelled sets still gate


def _lookup(tmp_path: Path, *reports: dict[str, Any] | str | None) -> Any:
    from disseqt_agent_testing.judges import JudgeConfig
    from disseqt_agent_testing.judges.calibration import lookup_calibration
    from disseqt_agent_testing.judges.kinds import get_kind

    dirs = []
    for n, report in enumerate(reports):
        root = tmp_path / f"cal{n}"
        dirs.append(str(root))
        if report is None:
            continue
        target = root / KIND / "idhash.json"
        target.parent.mkdir(parents=True)
        target.write_text(report if isinstance(report, str) else json.dumps(report))
    spec = get_kind(KIND)
    assert spec is not None
    return lookup_calibration(spec, "idhash", JudgeConfig(calibration_dirs=tuple(dirs)))


def _report(**changes: Any) -> dict[str, Any]:
    base = {"kind": KIND, "identity_hash": "idhash", "gating_allowed": True, "failures": []}
    return {**base, **changes}


def test_the_first_calibration_report_found_decides(tmp_path: Path) -> None:
    ok = _lookup(tmp_path / "a", None, _report(), _report(gating_allowed=False))
    assert (ok.gating, ok.status, ok.source) == ("gate", "calibrated", "project")
    failing = _lookup(tmp_path / "b", _report(gating_allowed=False), _report())
    assert (failing.gating, failing.status) == ("advisory", "failing")
    unreadable = _lookup(tmp_path / "c", "not json", _report())
    assert (unreadable.gating, unreadable.status) == ("advisory", "failing")
    assert _lookup(tmp_path / "d", None, None).status == "none"


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "decision"},
        {"identity_hash": "another"},
        {"failures": ["kappa 0.1 is below 0.6"]},
        {"failures": None},
        {"gating_allowed": "true"},
    ],
)
def test_a_report_must_match_its_kind_and_identity_and_list_no_failures(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    status = _lookup(tmp_path, _report(**changes))
    assert (status.gating, status.status) == ("advisory", "failing")
    for missing in ("kind", "identity_hash", "failures"):
        report = _report()
        del report[missing]
        assert _lookup(tmp_path / missing, report).gating == "advisory"


def test_cache_prune_and_verify_leave_files_the_store_did_not_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    from disseqt_agent_testing.judges import JsonDirStore

    key = "ab" + "c" * 62
    store = JsonDirStore(tmp_path)
    store.put("fake", "m", key, {"key": key, "response": {"text": "x"}})
    strangers = [
        tmp_path / "package.json",
        tmp_path / "web" / "tsconfig.json",
        tmp_path / "fake__m" / "ab" / "notes.json",
        tmp_path / "a" / "ab" / f"{key}.json",
        tmp_path / "fake__m" / "zz" / f"{key}.json",
    ]
    for stranger in strangers:
        stranger.parent.mkdir(parents=True, exist_ok=True)
        stranger.write_text('{"name": "mine"}')
    entry = store.path_for("fake", "m", key)
    for path in (*strangers, entry):
        os.utime(path, (0, 0))
    assert store.list() == [entry]
    assert judges_cli(["cache", "verify", "--dir", str(tmp_path)]) == 0
    assert judges_cli(["cache", "prune", "--dir", str(tmp_path), "--older-than-days", "1"]) == 0
    out = capsys.readouterr().out
    assert "1 entries pruned" in out and "package.json" not in out
    assert not entry.exists() and all(s.exists() for s in strangers)
