"""
Direct unit tests for the canonical tool-call adapters in
``_tool_calls.py``.

These are provider-independent — no wrapper glue, no HTTP mocks —
so they cover shape corner-cases that the per-provider integration
tests can miss.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from disseqt_agentic_sdk.instrumentation._tool_calls import (
    from_anthropic,
    from_gemini,
    from_openai,
)


class TestFromOpenAICustomTool:
    """
    Regression: OpenAI's "custom tool calling" feature returns
    ``{id, type: 'custom', custom: {name, input}}`` — no ``.function``.
    The old adapter checked only ``.function`` / ``.name`` at the top
    level, so custom tool calls came back as an empty list (no name)
    and were silently dropped. TP-2128 P1 #1.4.
    """

    def test_custom_tool_shape_recognized(self):
        tool_calls = [
            SimpleNamespace(
                id="call_custom_1",
                type="custom",
                function=None,
                custom=SimpleNamespace(name="run_python", input="print('hi')"),
            )
        ]
        got = from_openai(tool_calls)
        assert got == [
            {
                "id": "call_custom_1",
                "name": "run_python",
                "arguments": "print('hi')",
            }
        ]

    def test_function_tool_still_works(self):
        # Standard function tool call — the pre-fix common case.
        tool_calls = [
            SimpleNamespace(
                id="call_fn_1",
                type="function",
                function=SimpleNamespace(name="get_weather", arguments='{"loc":"Paris"}'),
                custom=None,
            )
        ]
        got = from_openai(tool_calls)
        assert got == [
            {
                "id": "call_fn_1",
                "name": "get_weather",
                "arguments": '{"loc":"Paris"}',
            }
        ]

    def test_mixed_function_and_custom_in_one_response(self):
        tool_calls = [
            SimpleNamespace(
                id="call_a",
                function=SimpleNamespace(name="get_weather", arguments='{"loc":"NYC"}'),
                custom=None,
            ),
            SimpleNamespace(
                id="call_b",
                function=None,
                custom=SimpleNamespace(name="run_sql", input="SELECT 1"),
            ),
        ]
        got = from_openai(tool_calls)
        assert len(got) == 2
        assert got[0]["name"] == "get_weather"
        assert got[1]["name"] == "run_sql"
        assert got[1]["arguments"] == "SELECT 1"

    def test_dict_shape_custom(self):
        # Providers that hand back plain dicts (LiteLLM proxies, some
        # test doubles) must work too.
        tool_calls = [
            {
                "id": "call_c",
                "type": "custom",
                "function": None,
                "custom": {"name": "generate_pdf", "input": "layout=A4"},
            }
        ]
        got = from_openai(tool_calls)
        assert got[0] == {
            "id": "call_c",
            "name": "generate_pdf",
            "arguments": "layout=A4",
        }


class TestFromGeminiIdChoice:
    """Related regression: prefer real FunctionCall.id over synthesized."""

    def test_real_id_preferred_when_present(self):
        parts = [
            SimpleNamespace(
                function_call=SimpleNamespace(
                    id="fc_real_123",
                    name="get_weather",
                    args={"loc": "Paris"},
                )
            )
        ]
        got = from_gemini(parts, response_id="resp-X")
        assert got[0]["id"] == "fc_real_123"

    def test_response_id_prefixed_when_no_real_id(self):
        parts = [
            SimpleNamespace(function_call=SimpleNamespace(id=None, name="get_weather", args={}))
        ]
        got = from_gemini(parts, response_id="resp-Y")
        assert got[0]["id"] == "resp-Y_call_0"

    def test_plain_fallback_without_response_id(self):
        parts = [SimpleNamespace(function_call=SimpleNamespace(id=None, name="x", args={}))]
        got = from_gemini(parts)
        assert got[0]["id"] == "call_0"


class TestFromAnthropicBasic:
    def test_tool_use_block_extracted(self):
        blocks = [
            SimpleNamespace(
                type="tool_use",
                id="toolu_1",
                name="get_weather",
                input={"loc": "Paris"},
            ),
            SimpleNamespace(type="text", text="here you go"),
        ]
        got = from_anthropic(blocks)
        assert len(got) == 1
        assert got[0]["id"] == "toolu_1"
        assert got[0]["name"] == "get_weather"
        assert json.loads(got[0]["arguments"]) == {"loc": "Paris"}

    def test_server_tool_use_block_captured(self):
        """
        TP-2128 Appendix: newer Anthropic responses can carry
        ``server_tool_use`` blocks (web_search, code_execution, ...)
        alongside the classic ``tool_use`` shape. Both share
        {id, name, input}; both belong in the canonical tool_calls
        list so validators see the model's full tool activity.
        """
        blocks = [
            SimpleNamespace(
                type="tool_use",
                id="toolu_user",
                name="get_weather",
                input={"loc": "Paris"},
            ),
            SimpleNamespace(
                type="server_tool_use",
                id="srvtool_1",
                name="web_search",
                input={"query": "capital of France"},
            ),
            # Result blocks are NOT tool CALLS — must be ignored so
            # tool-failure-rate doesn't double-count.
            SimpleNamespace(
                type="web_search_tool_result",
                tool_use_id="srvtool_1",
                content=[],
            ),
        ]
        got = from_anthropic(blocks)
        assert len(got) == 2, f"expected both tool_use and server_tool_use; got {got}"
        assert got[0]["name"] == "get_weather"
        assert got[1]["name"] == "web_search"
        assert got[1]["id"] == "srvtool_1"
        assert json.loads(got[1]["arguments"]) == {"query": "capital of France"}
