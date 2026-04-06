#!/usr/bin/env python3
"""Examples for the Agentic Behavior canonical layer.

Covers all 8 agentic validators. All use AgenticBehaviourRequest with:
  ``conversation_history`` — list of "role: message" strings
  ``tool_calls``           — list of tool/API call dicts
  ``agent_responses``      — list of agent reply strings
  ``reference_data``       — dict with validator-specific keys

Mandatory fields per config:
  Plan Coherence     : tool_calls
  Plan Optimality    : tool_calls + reference_data.ideal_plan_length
  Fallback Rate      : agent_responses
  Tool Failure Rate  : tool_calls  (each call must have a 'result' key)
  Tool Call Accuracy : tool_calls + reference_data.expected_tool_calls
  Topic Adherence    : conversation_history + reference_data.expected_topics
  Intent Resolution  : conversation_history + agent_responses + reference_data.reference_intent
  Agent Goal Accuracy: agent_responses + reference_data.target_goal
"""

from disseqt_sdk import Client, SDKConfigInput
from disseqt_sdk.models.agentic_behaviour import AgenticBehaviourRequest
from disseqt_sdk.validators.agentic_behavior.agent_goal_accuracy import AgentGoalAccuracyValidator
from disseqt_sdk.validators.agentic_behavior.fallback_rate import FallbackRateValidator
from disseqt_sdk.validators.agentic_behavior.intent_resolution import IntentResolutionValidator
from disseqt_sdk.validators.agentic_behavior.plan_coherence import PlanCoherenceValidator
from disseqt_sdk.validators.agentic_behavior.plan_optimality import PlanOptimalityValidator
from disseqt_sdk.validators.agentic_behavior.reliability import TopicAdherenceValidator
from disseqt_sdk.validators.agentic_behavior.tool_call_accuracy import ToolCallAccuracyValidator
from disseqt_sdk.validators.agentic_behavior.tool_failure_rate import ToolFailureRateValidator


def _run(client: Client, validator, label: str) -> None:
    print(f"\n=== {label} ===")
    try:
        result = client.validate(validator)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
CONVERSATION = [
    "user: I need to book a flight from London to Tokyo for next Monday.",
    "agent: I'll search for available flights from London to Tokyo on Monday.",
    "user: Make sure it's business class and under £3000.",
    "agent: I found several business class options. Let me check prices.",
]

TOOL_CALLS_GOOD = [
    {"tool": "search_flights", "args": {"from": "LHR", "to": "NRT", "date": "2026-02-02", "class": "business"}, "result": "success"},
    {"tool": "filter_by_price", "args": {"max_price": 3000, "currency": "GBP"}, "result": "success"},
    {"tool": "book_flight", "args": {"flight_id": "JL402", "passenger": "John Doe"}, "result": "success"},
]

TOOL_CALLS_WITH_FAILURES = [
    {"tool": "search_flights", "args": {"from": "LHR", "to": "NRT", "date": "2026-02-02"}, "result": "error: API timeout"},
    {"tool": "search_flights", "args": {"from": "LHR", "to": "NRT", "date": "2026-02-02"}, "result": "success"},
    {"tool": "get_prices", "args": {"flight_ids": ["JL402"]}, "result": "error: invalid flight ID"},
    {"tool": "book_flight", "args": {"flight_id": "JL402"}, "result": "success"},
]

AGENT_RESPONSES = [
    "I'll search for available business class flights from London to Tokyo on Monday.",
    "I found 3 business class options under £3000. The cheapest is Japan Airlines at £2,450.",
    "Your flight JL402 has been booked successfully. Confirmation number: ABC123.",
]


def main() -> None:
    """Run all Agentic Behavior examples."""
    client = Client(
        project_id="<ENTER_YOUR_PROJECT_ID_HERE>",
        api_key="<ENTER_YOUR_API_KEY_HERE>",
        base_url="http://localhost:8081",
        timeout=30,
    )

    # 1. Plan Coherence
    # Mandatory: tool_calls — evaluates logical consistency and structure of the tool plan.
    _run(client, PlanCoherenceValidator(
        data=AgenticBehaviourRequest(
            tool_calls=TOOL_CALLS_GOOD,
            conversation_history=CONVERSATION,
            agent_responses=AGENT_RESPONSES,
        ),
        config=SDKConfigInput(
            threshold=0.7,
            custom_labels=["Incoherent", "Partially Coherent", "Mostly Coherent", "Fully Coherent"],
            label_thresholds=[0.3, 0.5, 0.7],
        ),
    ), "1. Plan Coherence")

    # 2. Plan Optimality
    # Mandatory: tool_calls + reference_data.ideal_plan_length
    # Compares actual tool call count vs the ideal minimal plan.
    _run(client, PlanOptimalityValidator(
        data=AgenticBehaviourRequest(
            tool_calls=TOOL_CALLS_GOOD,
            conversation_history=CONVERSATION,
            agent_responses=AGENT_RESPONSES,
            reference_data={"ideal_plan_length": 3},
        ),
        config=SDKConfigInput(
            threshold=0.7,
            custom_labels=["Highly Suboptimal", "Suboptimal", "Near Optimal", "Optimal"],
            label_thresholds=[0.3, 0.5, 0.7],
        ),
    ), "2. Plan Optimality")

    # 3. Fallback Rate
    # Mandatory: agent_responses — measures how often the agent falls back to "I don't know".
    _run(client, FallbackRateValidator(
        data=AgenticBehaviourRequest(
            agent_responses=[
                "I'm sorry, I don't know how to do that.",
                "I found 3 business class flights under £3000.",
                "I don't have access to that information.",
                "Your booking is confirmed. Reference: ABC123.",
            ],
            conversation_history=CONVERSATION,
            tool_calls=TOOL_CALLS_GOOD,
        ),
        config=SDKConfigInput(
            threshold=0.7,
            custom_labels=["Always Fallback", "Often Fallback", "Occasional Fallback", "Rarely/Never Fallback"],
            label_thresholds=[0.3, 0.5, 0.7],
        ),
    ), "3. Fallback Rate")

    # 4. Tool Failure Rate
    # Mandatory: tool_calls (each must have a 'result' field indicating success/failure).
    _run(client, ToolFailureRateValidator(
        data=AgenticBehaviourRequest(
            tool_calls=TOOL_CALLS_WITH_FAILURES,
            conversation_history=CONVERSATION,
            agent_responses=AGENT_RESPONSES,
        ),
        config=SDKConfigInput(
            threshold=0.7,
            custom_labels=["Always Fails", "Often Fails", "Occasional Failure", "Rarely/Never Fails"],
            label_thresholds=[0.3, 0.5, 0.7],
        ),
    ), "4. Tool Failure Rate")

    # 5. Tool Call Accuracy
    # Mandatory: tool_calls + reference_data.expected_tool_calls
    # Evaluates whether the agent used the correct tools with the right arguments.
    _run(client, ToolCallAccuracyValidator(
        data=AgenticBehaviourRequest(
            tool_calls=TOOL_CALLS_GOOD,
            conversation_history=CONVERSATION,
            agent_responses=AGENT_RESPONSES,
            reference_data={
                "expected_tool_calls": [
                    {"tool": "search_flights", "args": {"from": "LHR", "to": "NRT", "class": "business"}},
                    {"tool": "filter_by_price", "args": {"max_price": 3000}},
                    {"tool": "book_flight"},
                ]
            },
        ),
        config=SDKConfigInput(
            threshold=0.7,
            custom_labels=["Always Incorrect", "Often Incorrect", "Occasional Error", "Mostly/Always Correct"],
            label_thresholds=[0.3, 0.5, 0.7],
        ),
    ), "5. Tool Call Accuracy")

    # 6. Topic Adherence
    # Mandatory: conversation_history + reference_data.expected_topics
    # Measures how well the agent stays on the expected topics throughout the conversation.
    _run(client, TopicAdherenceValidator(
        data=AgenticBehaviourRequest(
            conversation_history=CONVERSATION,
            agent_responses=AGENT_RESPONSES,
            tool_calls=TOOL_CALLS_GOOD,
            reference_data={
                "expected_topics": [
                    "flight booking",
                    "travel",
                    "London",
                    "Tokyo",
                    "business class",
                ]
            },
        ),
        config=SDKConfigInput(
            threshold=0.65,
            custom_labels=["Always Off-Topic", "Often Off-Topic", "Occasional Drift", "Mostly/Always On-Topic"],
            label_thresholds=[0.3, 0.5, 0.65],
        ),
    ), "6. Topic Adherence")

    # 7. Intent Resolution
    # Mandatory: conversation_history + agent_responses + reference_data.reference_intent
    # Assesses how well the agent identifies and resolves the user's underlying intent.
    _run(client, IntentResolutionValidator(
        data=AgenticBehaviourRequest(
            conversation_history=CONVERSATION,
            agent_responses=AGENT_RESPONSES,
            tool_calls=TOOL_CALLS_GOOD,
            reference_data={
                "reference_intent": "Book a business class flight from London to Tokyo on Monday for under £3000."
            },
        ),
        config=SDKConfigInput(
            threshold=0.65,
            custom_labels=["Incorrect Intent", "Partially Correct", "Mostly Correct", "Correct"],
            label_thresholds=[0.3, 0.5, 0.65],
        ),
    ), "7. Intent Resolution")

    # 8. Agent Goal Accuracy
    # Mandatory: agent_responses + reference_data.target_goal
    # Optional: reference_data.expected_actions
    # Measures how accurately the agent's actions/outputs align with the intended user goal.
    _run(client, AgentGoalAccuracyValidator(
        data=AgenticBehaviourRequest(
            agent_responses=AGENT_RESPONSES,
            conversation_history=CONVERSATION,
            tool_calls=TOOL_CALLS_GOOD,
            reference_data={
                "target_goal": "Successfully book a business class flight LHR→NRT on Monday under £3000.",
                "expected_actions": ["search flights", "filter by price", "confirm booking"],
            },
        ),
        config=SDKConfigInput(
            threshold=0.65,
            custom_labels=["Incorrect Goal", "Partially Correct", "Mostly Correct", "Correct"],
            label_thresholds=[0.5, 0.5, 0.65],
        ),
    ), "8. Agent Goal Accuracy")

    print("\n=== Agentic Behavior Examples Complete ===")


if __name__ == "__main__":
    main()
