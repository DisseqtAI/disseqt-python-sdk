"""
Regression tests for TP-2128 P2 #2.13.

The `pyproject.toml` extras use short human-friendly names ([gemini],
[mistral]) but the registry keys are the pip-install package names
(google-genai, mistralai). Before this fix, installing
`disseqt-ai-sdk[gemini]` and calling `instrument("gemini", client)`
returned False with an unknown_provider log and silently no-op'd.

The registry now carries an alias table so both names resolve to the
same instrumentor. `instrument_all(...)` still walks the canonical
table only, so each instrumentor is applied exactly once.
"""

from __future__ import annotations

import pytest

pytest.importorskip("google.genai")

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.instrumentation._registry import (  # noqa: E402
    INSTRUMENTOR_ALIASES,
    INSTRUMENTOR_CLASSES,
    resolve_provider_name,
)


class TestRegistryAliases:
    def test_aliases_point_to_canonical_keys(self):
        for alias, canonical in INSTRUMENTOR_ALIASES.items():
            assert (
                canonical in INSTRUMENTOR_CLASSES
            ), f"alias {alias!r} points to {canonical!r} which is not in INSTRUMENTOR_CLASSES"

    def test_resolve_passes_through_unknown_names(self):
        # Unknown names return themselves — resolver never fabricates keys.
        assert resolve_provider_name("not-a-provider") == "not-a-provider"

    def test_resolve_maps_extras_names_to_pip_names(self):
        assert resolve_provider_name("gemini") == "google-genai"
        assert resolve_provider_name("mistral") == "mistralai"

    def test_instrument_all_iterates_canonical_only(self):
        # instrument_all reads INSTRUMENTOR_CLASSES; aliases must NOT
        # appear there or we'd try to install the same instrumentor
        # twice per call. Guarded by an explicit assert on the two keys.
        assert "gemini" not in INSTRUMENTOR_CLASSES
        assert "mistral" not in INSTRUMENTOR_CLASSES

    def test_instrument_by_alias_reaches_the_gemini_instrumentor(self, recording_client):
        # Full round-trip: instrument("gemini", ...) succeeds and binds
        # the same active entry the canonical key would.
        from disseqt_agentic_sdk.instrumentation.auto import get_instrumented_client

        assert instrument("gemini", recording_client) is True
        try:
            # Both names refer to the same active instrumentor.
            assert get_instrumented_client("gemini") is recording_client
            assert get_instrumented_client("google-genai") is recording_client
        finally:
            assert uninstrument("gemini") is True
        # Cleanup path also resolves the alias.
        assert get_instrumented_client("google-genai") is None
