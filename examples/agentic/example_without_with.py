"""
Example: Creating traces and spans WITHOUT using 'with' statements.

This demonstrates manual span lifecycle management:
- Manual span creation
- Manual span ending
- Manual trace ending
- Manual trace sending
- Manual exception handling
- Manual parent-child relationship tracking (parent_span_id)

IMPORTANT: When not using 'with' statements, you MUST explicitly set
parent_span_id for child spans because:
1. When you call span.end(), it clears the context
2. Subsequent spans won't automatically detect the parent from context
3. You need to pass parent_span_id=parent_span.span_id explicitly
"""

import google.generativeai as genai

from disseqt_agentic_sdk import DisseqtAgenticClient
from disseqt_agentic_sdk.enums import SpanKind, SpanStatus
from disseqt_agentic_sdk.semantics import AgenticOperation
from disseqt_agentic_sdk.trace import DisseqtTrace

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

# Create trace manually (without 'with')
trace = DisseqtTrace(
    name="agent_workflow",
    org_id="org_123",
    project_id="proj_456",
    service_name="my-service",
    service_version="1.0.0",
    environment="production",
    intent_id="intent_123",
    workflow_id="workflow_456",
)

try:
    # SPAN 1: Root span - Agent execution (no parent, root=True)
    agent_span = trace.start_span("agent_execution", SpanKind.AGENT_EXEC)
    agent_span.set_agent_info("my_agent", "agent_001")
    agent_span.set_operation(AgenticOperation.INVOKE_AGENT)
    # Note: agent_span.span_id is automatically generated and set

    try:
        # SPAN 2: Child span - First Gemini LLM call
        # Explicitly set parent_span_id to agent_span.span_id
        llm_span_1 = trace.start_span(
            "gemini_chat_1",
            SpanKind.MODEL_EXEC,
            parent_span_id=agent_span.span_id,  # Explicitly set parent
        )
        llm_span_1.set_model_info("gemini-2.0-flash-exp", "google")
        llm_span_1.set_operation(AgenticOperation.CHAT)

        input_messages_1 = [{"role": "user", "content": "What's the weather in Paris?"}]
        llm_span_1.set_messages(input_messages=input_messages_1)

        try:
            # Make Gemini API call
            response_1 = model.generate_content("What's the weather in Paris?")

            output_messages_1 = [{"role": "assistant", "content": response_1.text}]
            llm_span_1.set_messages(output_messages=output_messages_1)
            llm_span_1.set_token_usage(
                input_tokens=response_1.usage_metadata.prompt_token_count,
                output_tokens=response_1.usage_metadata.candidates_token_count,
            )
            llm_span_1.set_status(SpanStatus.OK)
        except Exception as e:
            # Handle errors manually
            llm_span_1.set_error(str(e), error_type=type(e).__name__)
            llm_span_1.set_status(SpanStatus.ERROR, str(e))
        finally:
            # MUST manually end the span
            llm_span_1.end()

        # SPAN 3: Child span - Tool call (parent: agent_span)
        # Explicitly set parent_span_id to agent_span.span_id
        tool_span = trace.start_span(
            "weather_api_call",
            SpanKind.TOOL_EXEC,
            parent_span_id=agent_span.span_id,  # Explicitly set parent
        )
        tool_span.set_tool_info("get_weather", "tool_call_001")
        tool_span.set_operation(AgenticOperation.EXECUTE_TOOL)
        tool_span.set_attribute("tool.input.city", "Paris")

        try:
            # Simulate tool execution
            weather_data = {"temperature": "20°C", "condition": "sunny"}
            tool_span.set_attribute("tool.output.data", weather_data)
            tool_span.set_status(SpanStatus.OK)
        except Exception as e:
            tool_span.set_error(str(e), error_type=type(e).__name__)
            tool_span.set_status(SpanStatus.ERROR, str(e))
        finally:
            # MUST manually end the tool span
            tool_span.end()

        # SPAN 4: Child span - Second Gemini LLM call (parent: agent_span)
        # Explicitly set parent_span_id to agent_span.span_id
        llm_span_2 = trace.start_span(
            "gemini_chat_2",
            SpanKind.MODEL_EXEC,
            parent_span_id=agent_span.span_id,  # Explicitly set parent
        )
        llm_span_2.set_model_info("gemini-2.0-flash-exp", "google")
        llm_span_2.set_operation(AgenticOperation.CHAT)

        input_messages_2 = [{"role": "user", "content": "Summarize the weather info"}]
        llm_span_2.set_messages(input_messages=input_messages_2)

        try:
            # Make second Gemini API call
            response_2 = model.generate_content("Summarize the weather info")

            output_messages_2 = [{"role": "assistant", "content": response_2.text}]
            llm_span_2.set_messages(output_messages=output_messages_2)
            llm_span_2.set_token_usage(
                input_tokens=response_2.usage_metadata.prompt_token_count,
                output_tokens=response_2.usage_metadata.candidates_token_count,
            )
            llm_span_2.set_status(SpanStatus.OK)
        except Exception as e:
            llm_span_2.set_error(str(e), error_type=type(e).__name__)
            llm_span_2.set_status(SpanStatus.ERROR, str(e))
        finally:
            # MUST manually end the span
            llm_span_2.end()

    except Exception as e:
        # Handle errors for agent span
        agent_span.set_error(str(e), error_type=type(e).__name__)
        agent_span.set_status(SpanStatus.ERROR, str(e))
    finally:
        # MUST manually end the agent span
        agent_span.end()

except Exception as e:
    # Handle trace-level errors
    print(f"Error in trace: {e}")
finally:
    # MUST manually end the trace
    trace.end()

    # MUST manually send the trace
    client.send_trace(trace)
    print(f"Trace sent: {trace.trace_id} with {len(trace.spans)} spans")

# Cleanup
client.shutdown()
