"""Cache modes and format, single-flight, the network gate and the cost ledger (C-, N-, L-)."""

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.judges import (
    FakeJudgeModel,
    FakeSlow,
    JudgeConfigError,
    JudgeWarning,
    Price,
    allow_judge_requests,
    cache_key,
    make_judge,
    register_provider,
)
from disseqt_agent_testing.judges.cache import resolve_mode
from disseqt_agent_testing.judges.cache.store import JsonDirStore, safe_segment
from disseqt_agent_testing.judges.models import ExtraMissing
from disseqt_agent_testing.judges.models.registry import unregister_provider
from disseqt_agent_testing.judges.network import requests_allowed

from .helpers import decision_task, engine, label

pytestmark = pytest.mark.filterwarnings("ignore::disseqt_agent_testing.judges.api.JudgeWarning")


def files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*.json"))


def test_c01_record_then_replay_with_zero_provider_calls_and_zero_cost(tmp_path: Path) -> None:
    with engine(
        lambda r: label("ask"), cache_mode="record", cache_dir=str(tmp_path), cost_per_call=0.01
    ) as (eng, model):
        first = eng.judge(decision_task())
    assert len(model.calls) == 3 and first.cost == pytest.approx(0.03)
    written = files(tmp_path)
    assert len(written) == 3
    assert written[0].parent.parent.name.startswith("fake__fake-judge-1")
    text = written[0].read_text()
    assert text.endswith("\n") and text.startswith('{\n  "cache_schema_version"')
    entry = json.loads(text)
    assert list(entry) == sorted(entry) and {"key", "request", "response", "metadata"} <= set(entry)
    assert entry["key"] == written[0].stem and "api_key" not in text.lower()
    with engine([], cache_mode="replay", cache_dir=str(tmp_path)) as (eng, model):
        again = eng.judge(decision_task())
    assert not model.calls and again.cached is True and again.cost == 0.0
    assert (again.label, again.agreement) == (first.label, first.agreement)
    assert again.raw["recorded_cost_usd"] == pytest.approx(0.03)


def test_c02_a_replay_miss_is_an_error_and_writes_nothing(tmp_path: Path) -> None:
    with engine([], cache_mode="replay", cache_dir=str(tmp_path)) as (eng, model):
        v = eng.judge(decision_task())
    assert v.error_code == "cache_miss_offline" and not model.calls and not files(tmp_path)
    assert "no cached response for sample" in v.reason and "DISSEQT_JUDGE_CACHE=record" in v.reason


def test_c04_c05_refresh_overwrites_and_off_does_nothing(tmp_path: Path) -> None:
    with engine(lambda r: label("ask"), cache_mode="record", cache_dir=str(tmp_path)) as (eng, _):
        eng.judge(decision_task())
    before = {p: p.read_text() for p in files(tmp_path)}
    with engine(lambda r: label("answer"), cache_mode="refresh", cache_dir=str(tmp_path)) as (
        eng,
        model,
    ):
        v = eng.judge(decision_task())
    assert v.label == "answer" and len(model.calls) == 3
    assert {p: p.read_text() for p in files(tmp_path)} != before
    with engine(lambda r: label("ask"), cache_mode="off", cache_dir=str(tmp_path / "other")) as (
        eng,
        model,
    ):
        eng.judge(decision_task())
        eng.judge(decision_task())
    assert len(model.calls) == 6 and not (tmp_path / "other").exists()


def test_a_fake_model_under_auto_mode_never_touches_the_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with engine(lambda r: label("ask")) as (eng, _):
        eng.judge(decision_task())
    assert not (tmp_path / ".disseqt").exists()


def key_args(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "provider": "fake", "model": "m-1", "base_url": None,
        "messages": [{"role": "user", "content": "hi"}], "output_schema": {"type": "object"},
        "schema_name": "decision", "params": {"temperature": 0.7}, "sample_index": 0,
        "kind": "decision", "prompt_hash": "abc",
    }  # fmt: skip
    base.update(over)
    return base


def test_c06_the_key_reacts_to_content_and_not_to_operational_settings() -> None:
    base = cache_key(**key_args())
    assert cache_key(**key_args(params={"temperature": 0.0})) != base
    assert cache_key(**key_args(sample_index=1)) != base
    assert cache_key(**key_args(messages=[{"role": "user", "content": "hi!"}])) != base
    assert cache_key(**key_args(prompt_hash="abd")) != base
    assert (
        cache_key(**key_args(model="m-2")) != base
        and cache_key(**key_args(base_url="http://x")) != base
    )
    assert cache_key(**key_args()) == base


def test_c11_the_key_is_pinned_by_a_golden_value() -> None:
    assert (
        cache_key(**key_args())
        == "4fe22f05d993ede679dbc8d9cf7d95490df371a73beb7daf25aa24923e9a2f80"
    )


def test_c07_a_failed_write_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonDirStore(tmp_path)
    key = "ab" + "0" * 62

    def boom(*a: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.put("fake", "m", key, {"key": key, "response": {}})
    assert not list(tmp_path.rglob("*"))[1:] or all(p.is_dir() for p in tmp_path.rglob("*"))


def test_c08_sixteen_threads_on_one_task_make_three_provider_calls() -> None:
    with engine(lambda r: FakeSlow(0.4, label("ask"))) as (eng, model):
        barrier = threading.Barrier(16)
        out: list[Any] = []

        def go() -> None:
            barrier.wait()
            out.append(eng.judge(decision_task()))

        threads = [threading.Thread(target=go) for _ in range(16)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    assert len(model.calls) == 3 and {v.label for v in out} == {"ask"} and len(out) == 16


def test_c09_a_corrupt_entry_is_an_error_in_replay_and_a_warned_overwrite_in_record(
    tmp_path: Path,
) -> None:
    with engine(lambda r: label("ask"), cache_mode="record", cache_dir=str(tmp_path)) as (eng, _):
        eng.judge(decision_task())
    for p in files(tmp_path):
        p.write_text("{not json")
    with engine([], cache_mode="replay", cache_dir=str(tmp_path)) as (eng, _):
        assert eng.judge(decision_task()).error_code == "cache_corrupt"
    with pytest.warns(JudgeWarning, match="corrupt"):
        with engine(lambda r: label("ask"), cache_mode="record", cache_dir=str(tmp_path)) as (
            eng,
            _,
        ):
            assert eng.judge(decision_task()).label == "ask"
    assert all(json.loads(p.read_text())["key"] == p.stem for p in files(tmp_path))


def test_c10_model_drift_is_warned_about_in_record_mode(tmp_path: Path) -> None:
    with engine(
        lambda r: label("ask"),
        cache_mode="record",
        cache_dir=str(tmp_path),
        response_model_id="snap-1",
    ) as (eng, _):
        eng.judge(decision_task())
    victim = files(tmp_path)[0]
    victim.unlink()  # one sample must be re-asked live, by a model that now answers as snap-2
    with engine(
        lambda r: label("ask"),
        cache_mode="record",
        cache_dir=str(tmp_path),
        response_model_id="snap-2",
    ) as (eng, _):
        eng.judge(decision_task())
        assert any("model drift" in w for w in eng.warnings)


def test_c12_hostile_model_ids_cannot_escape_the_cache_directory(tmp_path: Path) -> None:
    assert "/" not in safe_segment("../../etc/passwd") and not safe_segment("..").startswith(".")
    store = JsonDirStore(tmp_path / "cache")
    path = store.path_for("p", "../../../../tmp/evil", "ab" + "0" * 62)
    assert (tmp_path / "cache").resolve() in path.resolve().parents


def test_c13_ci_defaults_to_replay_with_requests_disallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CI", "true")
    assert resolve_mode("auto") == "replay" and requests_allowed(None, "replay") is False
    monkeypatch.delenv("CI")
    assert resolve_mode("auto") == "record" and requests_allowed(None, "replay") is True
    monkeypatch.setenv("DISSEQT_JUDGE_CACHE", "refresh")
    assert resolve_mode("auto") == "refresh" and resolve_mode("off") == "off"
    monkeypatch.setenv("DISSEQT_ALLOW_JUDGE_REQUESTS", "0")
    assert requests_allowed(None, "record") is False and requests_allowed(True, "record") is True


def test_n01_n02_the_gate_nests_and_restores() -> None:
    assert requests_allowed(None, "record") is True
    with allow_judge_requests(False):
        assert requests_allowed(True, "record") is False
        with allow_judge_requests(True):
            assert requests_allowed(False, "record") is True
        assert requests_allowed(None, "record") is False
    assert requests_allowed(None, "record") is True


def test_a_networked_model_is_blocked_by_the_gate_but_a_fake_is_not() -> None:
    with engine(lambda r: label("ask"), networked=True, cache_mode="off") as (eng, model):
        with allow_judge_requests(False):
            v = eng.judge(decision_task())
        assert v.error_code == "network_forbidden" and not model.calls
        assert "live judge requests are disabled" in v.reason
        assert eng.judge(decision_task()).label == "ask"
    with engine(lambda r: label("ask")) as (eng, _):
        with allow_judge_requests(False):
            assert eng.judge(decision_task()).label == "ask"


def test_n03_replay_needs_no_provider_sdk_and_no_keys(tmp_path: Path) -> None:
    with engine(lambda r: label("ask"), cache_mode="record", cache_dir=str(tmp_path)) as (eng, _):
        eng.judge(decision_task())
    built: list[str] = []

    def factory(model_id: str, **_: Any) -> Any:
        built.append(model_id)
        raise ExtraMissing("openai")

    register_provider("fake", factory, override=True)
    try:
        eng = make_judge("fake:fake-judge-1", cache_mode="replay", cache_dir=str(tmp_path))
        v = eng.judge(decision_task())
        eng.close()
    finally:
        unregister_provider("fake")
    assert v.label == "ask" and v.cached and not built  # the provider object was never constructed


def test_n04_record_mode_with_a_missing_sdk_is_extra_missing(tmp_path: Path) -> None:
    def factory(model_id: str, **_: Any) -> Any:
        raise ExtraMissing("openai")

    register_provider("sdkless", factory)
    try:
        eng = make_judge("sdkless:model-1", cache_mode="record", cache_dir=str(tmp_path))
        v = eng.judge(decision_task())
        eng.close()
    finally:
        unregister_provider("sdkless")
    assert v.error_code == "extra_missing"
    assert "provider package missing; install 'disseqt-agent-testing[openai]'" in v.reason


def test_l01_max_calls_stops_further_attempts() -> None:
    with engine(lambda r: label("ask"), max_calls=2) as (eng, model):
        v = eng.judge(decision_task())
    assert (
        len(model.calls) == 2 and v.status == "ok" and v.samples == 2
    )  # quorum of 2 of 3 still decides
    with engine(lambda r: label("ask"), max_calls=1) as (eng, model):
        v = eng.judge(decision_task())
    assert v.error_code in ("budget_exceeded", "insufficient_votes") and len(model.calls) == 1


def test_l02_max_usd_is_checked_before_every_attempt() -> None:
    """Once a call has settled, its real cost is held for every later attempt, so the third
    sample (0.8 spent + 0.4 held) is refused instead of overshooting the cap."""
    price = Price(1.0, 1.0)
    options: dict[str, Any] = {
        "networked": True, "cache_mode": "off", "price": price, "cost_per_call": 0.4,
        "max_usd": 1.0, "max_concurrency": 1,
    }  # fmt: skip
    with engine(lambda r: label("ask"), **options) as (eng, model):
        first = eng.judge(decision_task())
        assert first.status == "ok" and eng.ledger.spent_usd == pytest.approx(0.8)
        refused = eng.judge(decision_task("another question?"))
        assert (
            refused.error_code == "budget_exceeded"
            and "budget exhausted: max_usd" in refused.reason
        )
        assert len(model.calls) == 2 and eng.ledger.spent_usd <= 1.0


def test_l03_an_unpriced_networked_model_cannot_have_a_dollar_cap() -> None:
    with pytest.raises(JudgeConfigError) as err:
        make_judge(FakeJudgeModel([], networked=True), max_usd=2.0)
    assert err.value.code == "price_unknown"
    make_judge(FakeJudgeModel([], networked=False), max_usd=2.0).close()  # fakes are exempt


def test_l04_cache_hits_cost_nothing_and_are_counted(tmp_path: Path) -> None:
    with engine(
        lambda r: label("ask"), cache_mode="record", cache_dir=str(tmp_path), cost_per_call=0.02
    ) as (eng, _):
        eng.judge(decision_task())
        assert eng.ledger.spent_usd == pytest.approx(0.06)
    with engine([], cache_mode="replay", cache_dir=str(tmp_path)) as (eng, _):
        eng.judge(decision_task())
        s = eng.ledger.summary()
        assert (
            s["spent_usd"] == 0.0 and s["cache_hits"] == 3 and s["saved_usd"] == pytest.approx(0.06)
        )
        assert eng.stats()["cache_hits"] == 3


def test_l05_the_per_evaluation_call_cap() -> None:
    with engine(lambda r: label("ask"), max_judge_calls_per_eval=2) as (eng, model):
        with eng.eval_scope():
            a, b = eng.judge(decision_task("one?")), eng.judge(decision_task("two?"))
            c = eng.judge(decision_task("three?"))
        assert a.status == b.status == "ok" and c.error_code == "budget_exceeded"
        assert "max_judge_calls_per_eval" in c.reason and len(model.calls) == 6
        with eng.eval_scope():  # a fresh evaluation gets a fresh budget
            assert eng.judge(decision_task("four?")).status == "ok"


def test_token_accounting_flows_into_the_verdict() -> None:
    with engine(lambda r: label("ask"), tokens=(100, 20)) as (eng, _):
        v = eng.judge(decision_task())
    assert v.tokens == 360 and eng.ledger.tokens == 360


def test_engine_stats_and_context_manager() -> None:
    with engine(lambda r: label("ask")) as (eng, _):
        eng.judge(decision_task())
        stats = eng.stats()
    assert stats["judge_calls"] == 1 and stats["samples"] == 3 and stats["pending"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        eng.judge(decision_task())


def test_concurrent_samples_cannot_overrun_max_usd() -> None:
    from disseqt_agent_testing.judges import KindConfig

    with engine(
        lambda r: FakeSlow(0.01, label("ask")), cost_per_call=0.08, max_usd=0.10,
        kinds={"decision": KindConfig(samples=5)},
    ) as (eng, model):  # fmt: skip
        v = eng.judge(decision_task())
        summary = eng.ledger.summary()
        assert summary["spent_usd"] == pytest.approx(0.08) and summary["spent_usd"] <= 0.10
        assert len(model.calls) == 1 and summary["calls"] == 1
        assert v.status == "error" and v.error_code == "budget_exceeded"
        assert "budget exhausted: max_usd" in v.reason


def test_concurrent_samples_within_the_cap_all_run() -> None:
    from disseqt_agent_testing.judges import KindConfig

    with engine(
        lambda r: FakeSlow(0.01, label("ask")), cost_per_call=0.01, max_usd=0.10,
        kinds={"decision": KindConfig(samples=5)},
    ) as (eng, model):  # fmt: skip
        v = eng.judge(decision_task())
        assert (v.status, v.samples) == ("ok", 5) and len(model.calls) == 5
        assert eng.ledger.summary()["spent_usd"] == pytest.approx(0.05)
        assert not any("exceeded max_usd" in w for w in eng.warnings)


def test_a_single_call_dearer_than_the_cap_is_reported() -> None:
    from disseqt_agent_testing.judges import KindConfig

    with engine(
        lambda r: label("ask"), cost_per_call=0.25, max_usd=0.10,
        kinds={"decision": KindConfig(samples=3)},
    ) as (eng, model):  # fmt: skip
        v = eng.judge(decision_task())
        assert len(model.calls) == 1 and eng.ledger.summary()["spent_usd"] == pytest.approx(0.25)
        assert v.error_code == "budget_exceeded"
        assert any("exceeded max_usd" in w for w in eng.warnings)


def test_the_ledger_holds_the_highest_cost_seen_for_a_model() -> None:
    from disseqt_agent_testing.judges import BudgetExceeded, CostLedger

    ledger = CostLedger(max_usd=1.0)
    first = ledger.try_reserve(0.01, "m")
    assert first is not None and ledger.try_reserve(0.01, "m") is None  # cost not known yet
    other = ledger.try_reserve(0.01, "other")
    assert other is not None  # another model is not held back by it
    ledger.release(other)
    ledger.settle(first, 0.4)
    second, third = ledger.try_reserve(0.01, "m"), ledger.try_reserve(0.01, "m")
    assert second is not None and third is None  # 0.4 spent + 0.4 held: no room for a third
    ledger.settle(second, 0.4)
    with pytest.raises(BudgetExceeded):
        ledger.try_reserve(0.01, "m")
    assert ledger.spent_usd == pytest.approx(0.8) and ledger.calls == 2


def test_a_gate_override_is_removed_by_identity_not_by_position() -> None:
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen: list[bool] = []

    def thread_a() -> None:
        with allow_judge_requests(True):
            a_in.set()
            b_in.wait(5)
        a_out.set()

    def thread_b() -> None:
        a_in.wait(5)
        with allow_judge_requests(False):
            b_in.set()
            a_out.wait(5)
            seen.append(requests_allowed(None, "record"))

    workers = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(5)
    assert seen == [False]
    assert requests_allowed(None, "record") is True  # nothing is left on the stack
