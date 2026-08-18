"""Tests for the DisseqtInstrumentor base class + auto module."""

from __future__ import annotations

from disseqt_agentic_sdk.instrumentation import (
    AVAILABLE_INSTRUMENTORS,
    instrument,
    instrument_all,
    uninstrument,
    uninstrument_all,
)
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
