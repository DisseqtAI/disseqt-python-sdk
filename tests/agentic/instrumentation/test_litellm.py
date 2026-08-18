"""Tests for the LiteLLM instrumentor."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("litellm")

import litellm  # noqa: E402

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


class TestLiteLLMCompletion:
    def test_records_span_with_dual_attrs(self, recording_client):
        """
        LiteLLM has a built-in `mock_response=` param that short-circuits
        the provider call and returns a synthetic ModelResponse — perfect
        for wiring up our instrumentation without hitting any real API.
        """
        instrument("litellm", recording_client)
        try:
            result = litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "capital of France?"}],
                mock_response="Paris.",
                api_key="fake",
            )
        finally:
            uninstrument("litellm")

        assert result.choices[0].message.content == "Paris."

        span = find_span(recording_client, "litellm.completion")
        attrs = json.loads(span.attributes_json)

        assert attrs[AgenticAttributes.REQUEST_MODEL] == "gpt-4o-mini"
        assert attrs[AgenticAttributes.PROVIDER_NAME] == "litellm"
        # LiteLLM populates response.model with the underlying model name.
        assert attrs[AgenticAttributes.RESPONSE_MODEL] == "gpt-4o-mini"
        assert attrs[AgenticAttributes.OUTPUT_MESSAGES] == [
            {"role": "assistant", "content": "Paris."}
        ]

        assert attrs[GenAIAttributes.SYSTEM] == "litellm"
        assert attrs[GenAIAttributes.OPERATION_NAME] == "chat"
