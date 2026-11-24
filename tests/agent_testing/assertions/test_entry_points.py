"""Third-party metrics load from the ``disseqt_agent_testing.metrics`` entry-point group."""

import threading
import time
from importlib import metadata
from typing import Any

import pytest

from disseqt_agent_testing.assertions import (
    BaseMetric,
    EvalConfig,
    MetricResult,
    evaluate,
    list_metrics,
)
from disseqt_agent_testing.assertions.metrics import registry, unregister_metric
from disseqt_agent_testing.models import Expect

from .helpers import run_of


class FakeEntryPoint:
    def __init__(self, name: str, load: Any) -> None:
        self.name, self._load = name, load

    def load(self) -> Any:
        return self._load()


def _plugin() -> type:
    @registry.register_metric("from_plugin", "3")
    class FromPlugin(BaseMetric):
        rank = 5000

        def applicable(self, expect: Expect, config: EvalConfig) -> bool:
            return "plugin" in expect.extra

        def evaluate(self, ctx: Any) -> MetricResult:
            return self.ok(ctx, 1.0, True)

    return FromPlugin


def test_entry_point_metrics_are_loaded_once_and_planned(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        calls.append(group)
        return [FakeEntryPoint("ok", _plugin), FakeEntryPoint("broken", lambda: 1 / 0)]

    monkeypatch.setattr(registry, "_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    try:
        names = [m.name for m in list_metrics()]
        assert "from_plugin" in names  # a broken plugin is skipped, not fatal
        list_metrics()
        assert calls == ["disseqt_agent_testing.metrics"]  # loaded once
        rep = evaluate(run_of(), Expect(extra={"plugin": True}))
        assert rep.check("from_plugin").version == "3"  # type: ignore[union-attr]
    finally:
        unregister_metric("from_plugin")
        monkeypatch.setattr(registry, "_ENTRY_POINTS_LOADED", True)


def test_a_failing_entry_point_lookup_never_breaks_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*, group: str) -> list[Any]:
        raise RuntimeError("broken metadata")

    monkeypatch.setattr(registry, "_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(metadata, "entry_points", boom)
    try:
        assert evaluate(run_of(captured=True), Expect(max_steps=1)).outcome == "passed"
    finally:
        monkeypatch.setattr(registry, "_ENTRY_POINTS_LOADED", True)


def test_a_second_thread_waits_for_a_slow_entry_point_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loading = threading.Event()

    def slow_plugin() -> type:
        loading.set()
        time.sleep(0.05)
        return _plugin()

    monkeypatch.setattr(registry, "_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(
        metadata, "entry_points", lambda *, group: [FakeEntryPoint("slow", slow_plugin)]
    )
    seen: dict[str, list[str]] = {}

    def planned(key: str) -> None:
        seen[key] = [m.name for m in list_metrics()]

    first = threading.Thread(target=planned, args=("first",))
    second = threading.Thread(target=planned, args=("second",))
    try:
        first.start()
        assert loading.wait(5)
        second.start()  # arrives while the plugin is still importing
        first.join(5)
        second.join(5)
        assert "from_plugin" in seen["first"] and seen["second"] == seen["first"]
    finally:
        unregister_metric("from_plugin")
        monkeypatch.setattr(registry, "_ENTRY_POINTS_LOADED", True)


def test_an_entry_point_that_reads_the_registry_while_loading_does_not_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def curious_plugin() -> type:
        list_metrics()
        return _plugin()

    def fake_entry_points(*, group: str) -> list[FakeEntryPoint]:
        calls.append(group)
        return [FakeEntryPoint("curious", curious_plugin)]

    monkeypatch.setattr(registry, "_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    try:
        assert "from_plugin" in [m.name for m in list_metrics()]
        assert len(calls) == 1
    finally:
        unregister_metric("from_plugin")
        monkeypatch.setattr(registry, "_ENTRY_POINTS_LOADED", True)
