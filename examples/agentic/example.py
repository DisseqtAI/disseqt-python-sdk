import google.generativeai as genai

from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.semantics import AgenticOperation

# Initialize SDK client
client = DisseqtAgenticClient(
    api_key="your-api-key",
    org_id="org_123",
    project_id="proj_456",
    service_name="my-service",
    endpoint="http://localhost:8080/v1/traces",
)

# Initialize Gemini
genai.configure(api_key="your-gemini-api-key")
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Create trace with multiple spans
with start_trace(client, name="agent_workflow") as trace:
    # SPAN 1: Root span - Agent execution (parent: None, root=True)
    with trace.start_span("agent_execution", SpanKind.AGENT_EXEC) as agent_span:
        agent_span.set_agent_info("my_agent", "agent_001")
        agent_span.set_operation(AgenticOperation.INVOKE_AGENT)

        # SPAN 2: Child span - First Gemini LLM call (parent: agent_execution, root=False)
        with trace.start_span("gemini_chat_1", SpanKind.MODEL_EXEC) as llm_span_1:
            llm_span_1.set_model_info("gemini-2.0-flash-exp", "google")
            llm_span_1.set_operation(AgenticOperation.CHAT)

            input_messages_1 = [{"role": "user", "content": "What's the weather in Paris?"}]
            llm_span_1.set_messages(input_messages=input_messages_1)

            # Make Gemini API call
            response_1 = model.generate_content("What's the weather in Paris?")

            output_messages_1 = [{"role": "assistant", "content": response_1.text}]
            llm_span_1.set_messages(output_messages=output_messages_1)
            llm_span_1.set_token_usage(
                input_tokens=response_1.usage_metadata.prompt_token_count,
                output_tokens=response_1.usage_metadata.candidates_token_count,
            )

        # SPAN 3: Child span - Tool call (parent: agent_execution, root=False)
        with trace.start_span("weather_api_call", SpanKind.TOOL_EXEC) as tool_span:
            tool_span.set_tool_info("get_weather", "tool_call_001")
            tool_span.set_operation(AgenticOperation.EXECUTE_TOOL)
            tool_span.set_attribute("tool.input.city", "Paris")

            # Simulate tool execution
            weather_data = {"temperature": "20°C", "condition": "sunny"}
            tool_span.set_attribute("tool.output.data", weather_data)

        # SPAN 4: Child span - Second Gemini LLM call (parent: agent_execution, root=False)
        with trace.start_span("gemini_chat_2", SpanKind.MODEL_EXEC) as llm_span_2:
            llm_span_2.set_model_info("gemini-2.0-flash-exp", "google")
            llm_span_2.set_operation(AgenticOperation.CHAT)

            input_messages_2 = [{"role": "user", "content": "Summarize the weather info"}]
            llm_span_2.set_messages(input_messages=input_messages_2)

            # Make second Gemini API call
            response_2 = model.generate_content("Summarize the weather info")

            output_messages_2 = [{"role": "assistant", "content": response_2.text}]
            llm_span_2.set_messages(output_messages=output_messages_2)
            llm_span_2.set_token_usage(
                input_tokens=response_2.usage_metadata.prompt_token_count,
                output_tokens=response_2.usage_metadata.candidates_token_count,
            )

# Trace is automatically sent when exiting the 'with' block
# All 4 spans (1 root + 3 children) will be sent together
client.shutdown()
