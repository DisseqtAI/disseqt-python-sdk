"""Tests for the Google Gemini (google-genai) instrumentor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from google import genai  # noqa: E402

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake_response():
    part = MagicMock(text="Paris.")
    content = MagicMock(parts=[part], role="model")
    candidate = MagicMock(content=content, finish_reason="STOP")
    usage = MagicMock(prompt_token_count=7, response_token_count=2, total_token_count=9)
    response = MagicMock(
        response_id="gemini-fake",
        model_version="gemini-2.0-flash-001",
        candidates=[candidate],
        usage_metadata=usage,
    )
    return response


class TestGeminiGenerate:
    def test_records_span_with_dual_attrs(self, recording_client):
        instrument("google-genai", recording_client)
        try:
            # The google.genai.Client tries to auth from env; provide a fake key.
            client = genai.Client(api_key="fake")
            fake = _fake_response()
            # Patch the underlying call — Models.generate_content ultimately hits an API method.
            with patch(
                "google.genai.models.Models._generate_content",
                return_value=fake,
                create=True,
            ):
                result = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents="capital of France?",
                )
        finally:
            uninstrument("google-genai")

        assert result.candidates[0].content.parts[0].text == "Paris."

        span = find_span(recording_client, "gemini.generate_content")
        attrs = json.loads(span.attributes_json)

        assert attrs[AgenticAttributes.REQUEST_MODEL] == "gemini-2.0-flash-001"
        assert attrs[AgenticAttributes.PROVIDER_NAME] == "google"
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 7
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 2
        assert attrs[AgenticAttributes.RESPONSE_ID] == "gemini-fake"
        assert attrs[AgenticAttributes.RESPONSE_MODEL] == "gemini-2.0-flash-001"
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "STOP"
        assert attrs[AgenticAttributes.OUTPUT_MESSAGES] == [{"role": "model", "content": "Paris."}]

        assert attrs[GenAIAttributes.SYSTEM] == "gemini"
        assert attrs[GenAIAttributes.OPERATION_NAME] == "generate_content"
