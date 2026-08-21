"""Tests for the DisseqtInstrumentor base class + auto module."""

from __future__ import annotations

import threading

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
