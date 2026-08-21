"""Tests for the DisseqtInstrumentor base class + auto module."""

from __future__ import annotations

import inspect
import threading

import openai
import wrapt

from disseqt_agentic_sdk.instrumentation import (
    AVAILABLE_INSTRUMENTORS,
    instrument,
    instrument_all,
    uninstrument,
    uninstrument_all,
)
from disseqt_agentic_sdk.instrumentation import auto as auto_module
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor


class TestBase:
    def test_version_gate_skips_when_package_missing(self, recording_client):
        class MissingPackage(DisseqtInstrumentor):
            package_name = "definitely-not-a-real-package-9x8x7"

            def _instrument(self) -> None:  # pragma: no cover - shouldn't run
                raise AssertionError("must not be called")

        assert MissingPackage().instrument(recording_client) is False

    def test_double_instrument_is_noop(self, recording_client):
        # openai is installed in the test env
        assert instrument("openai", recording_client) is True
        assert instrument("openai", recording_client) is False
        uninstrument("openai")

    def test_unknown_provider_returns_false(self, recording_client):
        assert instrument("does-not-exist", recording_client) is False

    def test_instrument_all_lists_installed(self, recording_client):
        installed = instrument_all(recording_client)
        assert "openai" in installed
        assert set(installed).issubset(set(AVAILABLE_INSTRUMENTORS))
        uninstrument_all()

    def test_uninstrument_drops_client_reference(self, recording_client):
        # After uninstrument(), the instrumentor must not keep the client
        # alive via wrapper closures — long-lived processes that repeatedly
        # instrument/uninstrument would otherwise accumulate clients.
        assert instrument("openai", recording_client) is True
        instrumentor = auto_module._ACTIVE["openai"]
        assert instrumentor._client is recording_client
        uninstrument("openai")
        assert instrumentor._client is None

    def test_partial_instrument_failure_rolls_back(self, recording_client):
        # If _instrument() raises after patching some methods, those patches
        # must be rolled back — otherwise the SDK is left partly wrapped
        # with nothing tracking the leftovers.
        Completions = openai.resources.chat.completions.Completions
        original_create = inspect.getattr_static(Completions, "create")

        class FlakyInstrumentor(DisseqtInstrumentor):
            package_name = "openai"

            def _instrument(self) -> None:
                self._wrap(
                    "openai.resources.chat.completions",
                    "Completions.create",
                    lambda w, i, a, kw: w(*a, **kw),
                )
                raise RuntimeError("boom halfway through")

        assert FlakyInstrumentor().instrument(recording_client) is False
        # The Completions.create attribute on the class must be the pristine
        # original — the failed instrumentor's wrapper was unwound.
        assert inspect.getattr_static(Completions, "create") is original_create

    def test_nested_wrap_by_another_lib_is_not_corrupted(self, recording_client):
        # Simulate another instrumentation library wrapping the same method
        # on top of ours. uninstrument() must not tear down their wrapper.
        Completions = openai.resources.chat.completions.Completions
        original_create = inspect.getattr_static(Completions, "create")

        assert instrument("openai", recording_client) is True

        def third_party(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
            return wrapped(*args, **kwargs)

        wrapt.wrap_function_wrapper(
            "openai.resources.chat.completions",
            "Completions.create",
            third_party,
        )
        top_after_third_party = inspect.getattr_static(Completions, "create")

        uninstrument("openai")

        # The third-party wrapper must still be installed as the outermost
        # layer — we did not touch it. If we had blindly rewritten the leaf
        # to its `__wrapped__`, this identity check would fail.
        assert inspect.getattr_static(Completions, "create") is top_after_third_party
        # Restore the pristine method so we don't leak wrappers to other tests.
        Completions.create = original_create

    def test_concurrent_instrument_calls_are_race_free(self, recording_client):
        # Many threads racing on the same provider must produce exactly one
        # successful instrument() and one registry entry — no duplicate
        # patching, no corrupted _ACTIVE.
        results: list[bool] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(16)

        def worker() -> None:
            barrier.wait()
            ok = instrument("openai", recording_client)
            with results_lock:
                results.append(ok)

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) == 1, f"expected exactly one winner, got {sum(results)}"
        assert "openai" in auto_module._ACTIVE
        uninstrument("openai")
        assert "openai" not in auto_module._ACTIVE
