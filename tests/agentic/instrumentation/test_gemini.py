"""Tests for the Google Gemini (google-genai) instrumentor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from google import genai  # noqa: E402
from google.genai import types as genai_types  # noqa: E402

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake_response():
    # NOTE: usage_metadata uses the real google-genai Pydantic type so a
    # future field-name typo (e.g. reading `response_token_count`, which
    # doesn't exist on this type) fails loudly instead of silently
    # returning a fresh MagicMock. See TP-2128 P0 #0.3 for the bug this
    # test now guards.
    part = MagicMock(text="Paris.")
    content = MagicMock(parts=[part], role="model")
    candidate = MagicMock(content=content, finish_reason="STOP")
    usage = genai_types.GenerateContentResponseUsageMetadata(
        prompt_token_count=7,
        candidates_token_count=2,
        total_token_count=9,
    )
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

    def test_reads_candidates_token_count_field(self, recording_client):
        """
        Regression guard for TP-2128 P0 #0.3.

        The instrumentor previously read ``response_token_count`` — a
        field that does not exist on
        ``GenerateContentResponseUsageMetadata`` (it only exists on the
        Live-API `GenerateContentResponse` type). This test uses a
        real-typed usage-metadata instance so a future rename to a
        non-existent field fails loudly at Pydantic construction time
        rather than silently returning 0 tokens.
        """
        instrument("google-genai", recording_client)
        try:
            client = genai.Client(api_key="fake")
            usage = genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=42,
                candidates_token_count=17,
                total_token_count=59,
            )
            part = MagicMock(text="hello")
            content = MagicMock(parts=[part], role="model")
            candidate = MagicMock(content=content, finish_reason="STOP")
            fake = MagicMock(
                response_id="gemini-tokens",
                model_version="gemini-2.0-flash-001",
                candidates=[candidate],
                usage_metadata=usage,
            )
            with patch(
                "google.genai.models.Models._generate_content",
                return_value=fake,
                create=True,
            ):
                client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents="x",
                )
        finally:
            uninstrument("google-genai")

        span = find_span(recording_client, "gemini.generate_content")
        attrs = json.loads(span.attributes_json)
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 42
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 17
        assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 42
        assert attrs[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 17
        assert attrs[GenAIAttributes.USAGE_TOTAL_TOKENS] == 59


class TestGeminiToolCalls:
    def test_captures_function_call_parts(self, recording_client):
        # Gemini surfaces tool calls as `candidates[0].content.parts[].function_call`
        # with a parsed `args` dict; the adapter synthesizes an id.
        instrument("google-genai", recording_client)
        try:
            client = genai.Client(api_key="fake")

            fc = MagicMock()
            fc.name = "get_weather"
            fc.args = {"location": "Paris"}
            # Function-call parts don't carry text; force to None so
            # MagicMock's auto-attribute doesn't accidentally look like text.
            part = MagicMock(function_call=fc, text=None)
            # A plain-text part alongside a function-call part is realistic.
            text_part = MagicMock(text="Let me check.", function_call=None)
            content = MagicMock(parts=[text_part, part], role="model")
            candidate = MagicMock(content=content, finish_reason="STOP")
            usage = genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=8,
                candidates_token_count=4,
                total_token_count=12,
            )
            fake = MagicMock(
                response_id="gemini-tools",
                model_version="gemini-2.0-flash-001",
                candidates=[candidate],
                usage_metadata=usage,
            )

            config = genai_types.GenerateContentConfig(
                tools=[
                    genai_types.Tool(
                        function_declarations=[
                            genai_types.FunctionDeclaration(
                                name="get_weather",
                                description="Get current weather",
                            )
                        ]
                    )
                ],
            )
            with patch(
                "google.genai.models.Models._generate_content",
                return_value=fake,
                create=True,
            ):
                client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents="weather in Paris?",
                    config=config,
                )
        finally:
            uninstrument("google-genai")

        span = find_span(recording_client, "gemini.generate_content")
        attrs = json.loads(span.attributes_json)

        req_tools = json.loads(attrs[AgenticAttributes.REQUEST_TOOLS])
        # google-genai's Tool model stringifies with its own repr; just
        # confirm the tool name is somewhere in the serialized payload.
        assert "get_weather" in json.dumps(req_tools)

        calls = attrs[AgenticAttributes.TOOL_CALLS]
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        # id is synthesized as f"{response_id}_call_{index}" so it stays
        # unique across multiple Gemini responses in the same agent_span
        # (TP-2128 P1 #1.3).
        assert calls[0]["id"] == "gemini-tools_call_0"


class TestGeminiLaneBCallIdCollision:
    def test_two_calls_in_one_agent_span_dont_collide(self, recording_client):
        """
        Regression: two Gemini responses inside a single ``agent_span``
        both synthesized ``call_0``, so the Lane-B aggregator's
        ``setdefault`` merge silently dropped the second call. Now the
        adapter namespaces synthesized ids by response_id, and both
        entries survive on the AGENT_EXEC span.
        """
        from disseqt_agentic_sdk import agent_span

        def _fake(tool_name: str, resp_id: str) -> MagicMock:
            fc = MagicMock()
            fc.name = tool_name
            fc.args = {}
            fc.id = None  # Gemini deployment that doesn't populate ids
            part = MagicMock(function_call=fc, text=None)
            content = MagicMock(parts=[part], role="model")
            candidate = MagicMock(content=content, finish_reason="STOP")
            return MagicMock(
                response_id=resp_id,
                model_version="gemini-2.0-flash-001",
                candidates=[candidate],
                usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=1,
                    candidates_token_count=1,
                    total_token_count=2,
                ),
            )

        instrument("google-genai", recording_client)
        try:
            client = genai.Client(api_key="fake")
            with agent_span(recording_client, "multi_tool_agent"):
                with patch(
                    "google.genai.models.Models._generate_content",
                    side_effect=[
                        _fake("get_weather", "resp-1"),
                        _fake("get_time", "resp-2"),
                    ],
                    create=True,
                ):
                    client.models.generate_content(
                        model="gemini-2.0-flash-001", contents="weather?"
                    )
                    client.models.generate_content(
                        model="gemini-2.0-flash-001", contents="what time?"
                    )
        finally:
            uninstrument("google-genai")

        agent = find_span(recording_client, "multi_tool_agent")
        attrs = json.loads(agent.attributes_json)
        calls = attrs[AgenticAttributes.TOOL_CALLS]
        # Before the fix: len == 1 (the second get_time dropped).
        # After: both survive because ids are namespaced by response_id.
        names = sorted(c["name"] for c in calls)
        assert names == [
            "get_time",
            "get_weather",
        ], f"expected both calls to survive; got {[c['name'] for c in calls]}"
        assert calls[0]["id"] != calls[1]["id"]
        # Each id is namespaced by its own response_id — no collision.
        assert "resp-1" in json.dumps(calls) and "resp-2" in json.dumps(calls)
