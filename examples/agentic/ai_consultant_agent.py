import base64
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

# Google ADK
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Disseqt SDK
from disseqt_agentic_sdk import DisseqtAgenticClient
from disseqt_agentic_sdk.context import get_current_trace
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.semantics import AgenticAttributes, AgenticOperation, AgenticProvider

# Define constants for the agent configuration
MODEL_ID = "gemini-2.5-flash"
APP_NAME = "ai_consultant_agent"
USER_ID = "consultant-user"
SESSION_ID = "consultant-session"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Disseqt SDK (set your endpoint and credentials)
# You can also use environment variables: DISSEQT_API_KEY, DISSEQT_ENDPOINT, etc.
DISSEQT_CLIENT: DisseqtAgenticClient | None = None
try:
    client = DisseqtAgenticClient(
        api_key=os.getenv("DISSEQT_API_KEY", "your-api-key"),
        org_id=os.getenv("DISSEQT_ORG_ID", "org_123"),
        project_id=os.getenv("DISSEQT_PROJECT_ID", "proj_456"),
        service_name=APP_NAME,
        service_version="1.0.0",
        environment=os.getenv("DISSEQT_ENV", "production"),
        endpoint=os.getenv("DISSEQT_ENDPOINT", "http://localhost:8080/v1/traces"),
    )
    DISSEQT_CLIENT = client
    logger.info("✅ Disseqt SDK initialized")
except Exception as e:
    logger.warning(f"⚠️ Disseqt SDK not initialized: {e}")
    DISSEQT_CLIENT = None


def sanitize_bytes_for_json(obj: Any) -> Any:
    """
    Recursively convert bytes objects to strings to ensure JSON serializability.

    Args:
        obj: Any object that might contain bytes

    Returns:
        Object with all bytes converted to strings
    """
    if isinstance(obj, bytes):
        try:
            # Try to decode as UTF-8 text first
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            # If not valid UTF-8, encode as base64 string
            return base64.b64encode(obj).decode("ascii")
    elif isinstance(obj, dict):
        return {key: sanitize_bytes_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_bytes_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_bytes_for_json(item) for item in obj)
    else:
        return obj


def safe_tool_wrapper(tool_func):
    """
    Wrapper to ensure tool functions never return bytes objects AND add tracing.

    Args:
        tool_func: The original tool function

    Returns:
        Wrapped function that sanitizes output and traces execution
    """

    def wrapped_tool(*args, **kwargs):
        # Get current trace from context
        trace = get_current_trace()
        if trace:
            # Create tool call span
            with trace.start_span(f"tool_{tool_func.__name__}", SpanKind.TOOL_EXEC) as span:
                span.set_tool_info(
                    tool_name=tool_func.__name__, tool_call_id=f"{tool_func.__name__}_{id(args)}"
                )
                span.set_operation(AgenticOperation.EXECUTE_TOOL)

                # Set tool input
                span.set_attribute(
                    "tool.input",
                    {
                        "args": str(args)[:500],  # Truncate for safety
                        "kwargs": {k: str(v)[:200] for k, v in kwargs.items()},
                    },
                )

                try:
                    result = tool_func(*args, **kwargs)
                    sanitized_result = sanitize_bytes_for_json(result)

                    # Set tool output
                    span.set_attribute(
                        "tool.output",
                        {"status": "success", "result_type": type(sanitized_result).__name__},
                    )

                    return sanitized_result
                except Exception as e:
                    logger.error(f"Error in tool {tool_func.__name__}: {e}")
                    span.set_error(error_message=str(e), error_type=type(e).__name__)
                    return {
                        "error": f"Tool execution failed: {str(e)}",
                        "tool": tool_func.__name__,
                        "status": "error",
                    }
        else:
            # No trace context, just execute normally
            try:
                result = tool_func(*args, **kwargs)
                return sanitize_bytes_for_json(result)
            except Exception as e:
                logger.error(f"Error in tool {tool_func.__name__}: {e}")
                return {
                    "error": f"Tool execution failed: {str(e)}",
                    "tool": tool_func.__name__,
                    "status": "error",
                }

    # Preserve function metadata
    wrapped_tool.__name__ = tool_func.__name__
    wrapped_tool.__doc__ = tool_func.__doc__
    return wrapped_tool


@dataclass
class MarketInsight:
    """Structure for market research insights"""

    category: str
    finding: str
    confidence: float
    source: str


def analyze_market_data(research_query: str, industry: str = "") -> dict[str, Any]:
    """
    Analyze market data and generate insights

    Args:
        research_query: The business query to analyze
        industry: Optional industry context

    Returns:
        Market analysis insights and recommendations
    """
    # Simulate market analysis - in real implementation this would process actual search results
    insights = []

    if "startup" in research_query.lower() or "launch" in research_query.lower():
        insights.extend(
            [
                MarketInsight(
                    "Market Opportunity",
                    "Growing market with moderate competition",
                    0.8,
                    "Market Research",
                ),
                MarketInsight(
                    "Risk Assessment",
                    "Standard startup risks apply - funding, competition",
                    0.7,
                    "Analysis",
                ),
                MarketInsight(
                    "Recommendation",
                    "Conduct MVP testing before full launch",
                    0.9,
                    "Strategic Planning",
                ),
            ]
        )

    if "saas" in research_query.lower() or "software" in research_query.lower():
        insights.extend(
            [
                MarketInsight(
                    "Technology Trend",
                    "Cloud-based solutions gaining adoption",
                    0.9,
                    "Tech Analysis",
                ),
                MarketInsight(
                    "Customer Behavior",
                    "Businesses prefer subscription models",
                    0.8,
                    "Market Study",
                ),
            ]
        )

    if industry:
        insights.append(
            MarketInsight(
                "Industry Specific",
                f"{industry} sector shows growth potential",
                0.7,
                "Industry Report",
            )
        )

    return {
        "query": research_query,
        "industry": industry,
        "insights": [
            {
                "category": insight.category,
                "finding": insight.finding,
                "confidence": insight.confidence,
                "source": insight.source,
            }
            for insight in insights
        ],
        "summary": f"Analysis completed for: {research_query}",
        "total_insights": len(insights),
    }


def generate_strategic_recommendations(analysis_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generate strategic business recommendations based on analysis

    Args:
        analysis_data: Market analysis results

    Returns:
        List of strategic recommendations
    """
    recommendations = []

    # Generate recommendations based on insights
    insights = analysis_data.get("insights", [])

    if any("startup" in insight["finding"].lower() for insight in insights):
        recommendations.append(
            {
                "category": "Market Entry Strategy",
                "priority": "High",
                "recommendation": "Implement phased market entry with MVP testing",
                "rationale": "Reduces risk and validates market fit before major investment",
                "timeline": "3-6 months",
                "action_items": [
                    "Develop minimum viable product",
                    "Identify target customer segment",
                    "Conduct market validation tests",
                ],
            }
        )

    if any("saas" in insight["finding"].lower() for insight in insights):
        recommendations.append(
            {
                "category": "Technology Strategy",
                "priority": "Medium",
                "recommendation": "Focus on cloud-native architecture and subscription model",
                "rationale": "Aligns with market trends and customer preferences",
                "timeline": "2-4 months",
                "action_items": [
                    "Design scalable cloud infrastructure",
                    "Implement subscription billing system",
                    "Plan for multi-tenant architecture",
                ],
            }
        )

    # Always include risk management
    recommendations.append(
        {
            "category": "Risk Management",
            "priority": "High",
            "recommendation": "Establish comprehensive risk monitoring framework",
            "rationale": "Proactive risk management is essential for business success",
            "timeline": "1-2 months",
            "action_items": [
                "Identify key business risks",
                "Develop mitigation strategies",
                "Implement monitoring systems",
            ],
        }
    )

    return recommendations


def perplexity_search(
    query: str,
    system_prompt: str = "Be precise and concise. Focus on business insights and market data.",
) -> dict[str, Any]:
    """Search the web using Perplexity AI for real-time information and insights."""
    # Get current trace for LLM call tracing
    trace = get_current_trace()

    try:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            return {
                "error": "Perplexity API key not found. Please set PERPLEXITY_API_KEY environment variable.",
                "query": query,
                "status": "error",
            }

        # Trace the Perplexity LLM call
        if trace:
            with trace.start_span("perplexity_llm_call", SpanKind.MODEL_EXEC) as span:
                span.set_model_info(model_name="sonar", provider=AgenticProvider.PERPLEXITY)
                span.set_operation(AgenticOperation.CHAT)

                input_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ]
                span.set_messages(input_messages=input_messages)

                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    json={"model": "sonar", "messages": input_messages},
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

                if "choices" in result and result["choices"]:
                    output_messages = [
                        {"role": "assistant", "content": result["choices"][0]["message"]["content"]}
                    ]
                    span.set_messages(output_messages=output_messages)

                    # Set token usage
                    usage = result.get("usage", {})
                    if usage:
                        span.set_token_usage(
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                        )

                    # Set response attributes
                    span.set_attribute(AgenticAttributes.RESPONSE_ID, result.get("id", ""))
                    span.set_attribute(
                        AgenticAttributes.RESPONSE_MODEL, result.get("model", "sonar")
                    )

                    return {
                        "query": query,
                        "content": result["choices"][0]["message"]["content"],
                        "citations": result.get("citations", []),
                        "search_results": result.get("search_results", []),
                        "status": "success",
                        "source": "Perplexity AI",
                        "model": result.get("model", "sonar"),
                        "usage": result.get("usage", {}),
                        "response_id": result.get("id", ""),
                        "created": result.get("created", 0),
                    }
                else:
                    span.set_error("No response content found", "APIError")
                    return {
                        "error": "No response content found",
                        "query": query,
                        "status": "error",
                        "raw_response": result,
                    }
        else:
            # No trace context, execute normally
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                json={
                    "model": "sonar",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                },
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and result["choices"]:
                return {
                    "query": query,
                    "content": result["choices"][0]["message"]["content"],
                    "citations": result.get("citations", []),
                    "search_results": result.get("search_results", []),
                    "status": "success",
                    "source": "Perplexity AI",
                    "model": result.get("model", "sonar"),
                    "usage": result.get("usage", {}),
                    "response_id": result.get("id", ""),
                    "created": result.get("created", 0),
                }
            return {
                "error": "No response content found",
                "query": query,
                "status": "error",
                "raw_response": result,
            }
    except Exception as e:
        if trace:
            # Error will be caught by tool wrapper span
            pass
        return {"error": f"Error: {str(e)}", "query": query, "status": "error"}


# Define the consultant tools with safety wrappers
consultant_tools = [
    safe_tool_wrapper(analyze_market_data),
    safe_tool_wrapper(generate_strategic_recommendations),
    safe_tool_wrapper(perplexity_search),
]

INSTRUCTIONS = """You are a senior AI business consultant specializing in market analysis and strategic planning.

Your expertise includes:
- Business strategy development and recommendations
- Risk assessment and mitigation planning
- Implementation planning with timelines
- Market analysis using your knowledge and available tools
- Real-time web research using Perplexity AI search capabilities

When consulting with clients:
1. Use Perplexity search to gather current market data, competitor information, and industry trends from the web
2. Use the market analysis tool to process business queries and generate insights
3. Use the strategic recommendations tool to create actionable business advice
4. Provide clear, specific recommendations with implementation timelines
5. Focus on practical solutions that drive measurable business outcomes

**Core Responsibilities:**
- Conduct real-time web research using Perplexity AI for current market data and trends
- Analyze competitive landscapes and market opportunities using search results and your knowledge
- Provide strategic guidance with clear action items based on up-to-date information
- Assess risks and suggest mitigation strategies using current market conditions
- Create implementation roadmaps with realistic timelines
- Generate comprehensive business insights combining web research with analysis tools

**Critical Rules:**
- Always search for current market data, trends, and competitor information when relevant using Perplexity search
- Base recommendations on sound business principles, current market insights, and real-time web data
- Provide specific, actionable advice rather than generic guidance
- Include timelines and success metrics in recommendations
- Prioritize recommendations by business impact and feasibility
- Use Perplexity search to validate assumptions and gather supporting evidence with citations
- Combine search results with your analysis tools for comprehensive consultation

**Search Strategy:**
- Use Perplexity search for competitor analysis, market size, industry trends, and regulatory changes
- Look up recent news, funding rounds, and market developments in relevant sectors
- Verify market assumptions with current web data before making recommendations
- Research best practices and case studies from similar businesses
- Always include citations and sources when referencing search results

Always maintain a professional, analytical approach while being results-oriented.
Use all available tools including Perplexity search to provide comprehensive, well-researched consultation backed by current web data and citations."""


# ADK Callbacks for tracing LLM calls
def before_model_callback(
    callback_context: CallbackContext, llm_request
) -> types.Content | None:
    """Callback before LLM request - start LLM span"""
    trace = get_current_trace()
    if trace:
        # Extract model info
        model = getattr(llm_request, "model", MODEL_ID)

        # Extract prompt from request
        prompt_text = ""
        if hasattr(llm_request, "contents") and llm_request.contents:
            for content in llm_request.contents:
                if hasattr(content, "parts") and content.parts:
                    for part in content.parts:
                        if hasattr(part, "text") and part.text:
                            prompt_text += part.text + "\n"

        # Create LLM span and store in context state
        span = trace.start_span(f"llm_{model}", SpanKind.MODEL_EXEC)
        span.set_model_info(model_name=model, provider=AgenticProvider.GOOGLE)
        span.set_operation(AgenticOperation.CHAT)

        if prompt_text:
            span.set_messages(input_messages=[{"role": "user", "content": prompt_text[:1000]}])

        # Store span in callback context state
        callback_context.state.update(
            {"llm_span": span, "llm_model": model, "llm_prompt": prompt_text[:500]}
        )

    return None  # Allow normal execution


def after_model_callback(
    callback_context: CallbackContext, llm_response
) -> types.Content | None:
    """Callback after LLM response - complete LLM span"""
    state = callback_context.state.to_dict()
    span = state.get("llm_span")

    if span:
        # Extract response
        response_text = ""
        if hasattr(llm_response, "text"):
            response_text = llm_response.text
        elif hasattr(llm_response, "content") and llm_response.content:
            for content in llm_response.content:
                if hasattr(content, "parts") and content.parts:
                    for part in content.parts:
                        if hasattr(part, "text") and part.text:
                            response_text += part.text + "\n"

        if response_text:
            span.set_messages(
                output_messages=[{"role": "assistant", "content": response_text[:1000]}]
            )

        # Extract usage if available
        if hasattr(llm_response, "usage"):
            usage = llm_response.usage
            if hasattr(usage, "prompt_token_count") and hasattr(usage, "candidates_token_count"):
                span.set_token_usage(
                    input_tokens=usage.prompt_token_count,
                    output_tokens=usage.candidates_token_count,
                )

        # End the span
        span.end()

    return None  # Allow normal execution


# Define the agent instance with callbacks
root_agent = LlmAgent(
    model=MODEL_ID,
    name=APP_NAME,
    description="An AI business consultant that provides market research, strategic analysis, and actionable recommendations.",
    instruction=INSTRUCTIONS,
    tools=consultant_tools,
    output_key="consultation_response",
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)

# Setup Runner and Session Service
session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


def run_with_tracing(user_input: str, session_id: str = SESSION_ID):
    """
    Run the agent with tracing enabled.

    Args:
        user_input: User's query/input
        session_id: Session ID for the conversation

    Returns:
        Agent response
    """
    from disseqt_agentic_sdk import start_trace

    if DISSEQT_CLIENT is None:
        logger.warning("Disseqt SDK not initialized, skipping tracing")
        return None

    # Create trace for the entire agent execution
    with start_trace(
        DISSEQT_CLIENT,
        name="ai_consultant_agent",
        intent_id=f"consultation_{session_id}",
        workflow_id="business_consultation",
        user_id=USER_ID,
    ) as trace:
        # Root span for agent execution
        with trace.start_span("agent_execution", SpanKind.AGENT_EXEC) as agent_span:
            agent_span.set_agent_info(agent_name=APP_NAME, agent_id=APP_NAME, agent_version="1.0.0")
            agent_span.set_operation(AgenticOperation.INVOKE_AGENT)
            agent_span.set_attribute("user_input", user_input[:500])
            agent_span.set_attribute("session_id", session_id)

            # Run the agent (LLM calls and tool calls will be traced automatically)
            result = runner.run(user_input, session_id=session_id)

            # Set agent output
            if hasattr(result, "content"):
                agent_span.set_attribute("agent_output", str(result.content)[:500])
            elif isinstance(result, dict):
                agent_span.set_attribute(
                    "agent_output", str(result.get("consultation_response", ""))[:500]
                )

            return result


if __name__ == "__main__":
    print("🤖 AI Consultant Agent with Google ADK + Disseqt Tracing")
    print("=========================================================")
    print()
    print("This agent provides comprehensive business consultation including:")
    print("• Market research and analysis")
    print("• Strategic recommendations")
    print("• Implementation planning")
    print("• Risk assessment")
    print()
    print("✅ Tracing Enabled:")
    print("• Agent execution spans")
    print("• LLM call spans (via ADK callbacks)")
    print(
        "• Tool call spans (analyze_market_data, generate_strategic_recommendations, perplexity_search)"
    )
    print()
    print("To use this agent:")
    print("1. Set environment variables:")
    print("   export DISSEQT_API_KEY='your-key'")
    print("   export DISSEQT_ENDPOINT='http://localhost:8080/v1/traces'")
    print("   export PERPLEXITY_API_KEY='your-perplexity-key'")
    print("2. Run: adk web .")
    print("3. Open the web interface")
    print("4. Select 'AI Business Consultant' agent")
    print("5. Start your consultation")
    print()
    print("Or use programmatically with tracing:")
    print("  result = run_with_tracing('I want to launch a SaaS startup')")
    print()
    print("Example queries:")
    print('• "I want to launch a SaaS startup for small businesses"')
    print('• "Should I expand my retail business to e-commerce?"')
    print('• "What are the market opportunities in the healthcare tech space?"')
    print()
    print("📊 Use the Eval tab in ADK web to save and evaluate consultation sessions!")
    print("📊 Traces are automatically sent to Disseqt backend!")
    print()
    print(f"✅ Agent '{APP_NAME}' initialized successfully!")
    print(f"   Model: {MODEL_ID}")
    print(f"   Tools: {len(consultant_tools)} available")
    print(f"   Session Service: {type(session_service).__name__}")
    print(f"   Runner: {type(runner).__name__}")
    print()

    # Example: Run with tracing
    if os.getenv("RUN_EXAMPLE"):
        print("Running example with tracing...")
        result = run_with_tracing("I want to launch a SaaS startup for small businesses")
        print(f"Result: {result}")
