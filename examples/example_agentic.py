#!/usr/bin/env python3
"""Example usage of Disseqt SDK - Agentic Behavior Validators.

This example demonstrates all agentic behavior validators available in the SDK.
These validators are designed to evaluate AI agent behaviors, tool usage,
planning capabilities, and conversation management.
"""

import logging
import sys

from disseqt_sdk import Client, SDKConfigInput
from disseqt_sdk.models.agentic_behaviour import AgenticBehaviourRequest
from disseqt_sdk.validators.agentic_behavior import (
    AgentGoalAccuracyValidator,
    FallbackRateValidator,
    IntentResolutionValidator,
    PlanCoherenceValidator,
    PlanOptimalityValidator,
    ToolCallAccuracyValidator,
    ToolFailureRateValidator,
    TopicAdherenceValidator,
)

# Configure logging for real-time output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run_validator(client: Client, name: str, validator) -> None:
    """Run a validator and print results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        result = client.validate(validator)
        print(f"✓ Result: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")


def main() -> None:
    """Demonstrate all agentic behavior validators."""
    # Initialize client (replace with your actual credentials)
    client = Client(
        project_id="e14f79b1-c839-44dc-96c8-f1166ed45a63",
        api_key="3a2e8d8b-3bd1-4300-a13a-58122f49b0a6",
        base_url="http://localhost:8081",
        timeout=30,
    )

    print("\n" + "=" * 60)
    print("  DISSEQT SDK - Agentic Behavior Validators Demo")
    print("=" * 60)

    # # =========================================================================
    # # 1. Topic Adherence Validator
    # # =========================================================================
    # # Evaluates if the agent stays on topic during conversations
    # topic_validator = TopicAdherenceValidator(
    #     data=AgenticBehaviourRequest(
    #         conversation_history=[
    #             "user: Tell me about machine learning algorithms.",
    #             "agent: Machine learning algorithms are computational methods that allow systems to learn from data.",
    #             "user: Can you explain neural networks?",
    #             "agent: Neural networks are a subset of machine learning inspired by the human brain.",
    #         ],
    #         tool_calls=[],
    #         agent_responses=[
    #             "Machine learning algorithms are computational methods that allow systems to learn from data.",
    #             "Neural networks are a subset of machine learning inspired by the human brain.",
    #         ],
    #         reference_data={
    #             "expected_topics": [
    #                 "machine learning",
    #                 "algorithms",
    #                 "neural networks",
    #                 "artificial intelligence",
    #             ]
    #         },
    #     ),
    #     config=SDKConfigInput(
    #         threshold=0.8,
    #         custom_labels=[
    #             "Always Off-Topic",
    #             "Often Off-Topic",
    #             "Occasional Drift",
    #             "Mostly On-Topic",
    #         ],
    #         label_thresholds=[0.6, 0.75, 0.85],
    #     ),
    # )
    # run_validator(client, "Topic Adherence", topic_validator)

    # # =========================================================================
    # # 2. Tool Call Accuracy Validator
    # # =========================================================================
    # # Evaluates accuracy of tool/function calls made by the agent
    # tool_accuracy_validator = ToolCallAccuracyValidator(
    #     data=AgenticBehaviourRequest(
    #         conversation_history=[
    #             "user: What's the weather in New York?",
    #             "agent: Let me check the weather for you.",
    #         ],
    #         tool_calls=[
    #             {
    #                 "name": "weather_api",
    #                 "args": {"location": "New York", "date": "2024-01-15"},
    #                 "result": {"temperature": 22, "condition": "sunny"},
    #             },
    #             {
    #                 "name": "format_response",
    #                 "args": {"style": "detailed"},
    #                 "result": {"formatted": True},
    #             },
    #         ],
    #         agent_responses=[
    #             "The current weather in New York is 22°C and sunny."
    #         ],
    #         reference_data={
    #             "expected_tool_calls": [
    #                 {
    #                     "name": "weather_api",
    #                     "args": {"location": "New York", "date": "2024-01-15"},
    #                 },
    #                 {
    #                     "name": "format_response",
    #                     "args": {"style": "detailed"},
    #                 },
    #             ]
    #         },
    #     ),
    #     config=SDKConfigInput(
    #         threshold=0.9,
    #         custom_labels=["Poor", "Fair", "Good", "Excellent"],
    #         label_thresholds=[0.5, 0.7, 0.9],
    #     ),
    # )
    # run_validator(client, "Tool Call Accuracy", tool_accuracy_validator)

    # # =========================================================================
    # # 3. Tool Failure Rate Validator
    # # =========================================================================
    # # Monitors the rate of tool call failures
    # tool_failure_validator = ToolFailureRateValidator(
    #     data=AgenticBehaviourRequest(
    #         conversation_history=[
    #             "user: Search for flights to Paris.",
    #             "agent: I'll search for available flights.",
    #         ],
    #         tool_calls=[
    #             {
    #                 "name": "search_flights",
    #                 "arguments": {"destination": "Paris", "date": "2024-03-15"},
    #                 "status": "success",
    #                 "result": {"flights": [{"airline": "Air France", "price": 450}]},
    #             },
    #             {
    #                 "name": "check_availability",
    #                 "arguments": {"flight_id": "AF123"},
    #                 "status": "failed",
    #                 "error": "Service temporarily unavailable",
    #             },
    #             {
    #                 "name": "get_price_details",
    #                 "arguments": {"flight_id": "AF123"},
    #                 "status": "success",
    #                 "result": {"total": 450, "taxes": 50},
    #             },
    #         ],
    #         agent_responses=["I found flights to Paris. There's an Air France flight for $450."],
    #         reference_data={"acceptable_failure_rate": 0.1},
    #     ),
    #     config=SDKConfigInput(
    #         threshold=0.9,
    #         custom_labels=["High Failure Rate", "Moderate", "Low", "Excellent"],
    #         label_thresholds=[0.7, 0.85, 0.95],
    #     ),
    # )
    # run_validator(client, "Tool Failure Rate", tool_failure_validator)

    # # =========================================================================
    # # 4. Agent Goal Accuracy Validator
    # # =========================================================================
    # # Measures how well the agent achieves the stated goal
    # goal_accuracy_validator = AgentGoalAccuracyValidator(
    #     data=AgenticBehaviourRequest(
    #         conversation_history=[
    #             "user: Help me book a restaurant for 4 people tonight at 7pm in downtown.",
    #             "agent: I'll help you find a restaurant. What cuisine do you prefer?",
    #             "user: Italian would be great.",
    #             "agent: I found 'La Bella Italia' downtown. They have availability for 4 at 7pm. Shall I book it?",
    #             "user: Yes, please book it.",
    #             "agent: Done! Your reservation at La Bella Italia is confirmed for 4 people at 7pm tonight.",
    #         ],
    #         tool_calls=[
    #             {
    #                 "name": "search_restaurants",
    #                 "arguments": {
    #                     "cuisine": "Italian",
    #                     "location": "downtown",
    #                     "party_size": 4,
    #                     "time": "19:00",
    #                 },
    #                 "result": {"restaurants": [{"name": "La Bella Italia", "available": True}]},
    #             },
    #             {
    #                 "name": "make_reservation",
    #                 "arguments": {
    #                     "restaurant": "La Bella Italia",
    #                     "party_size": 4,
    #                     "time": "19:00",
    #                 },
    #                 "result": {"confirmation": "RES-12345", "status": "confirmed"},
    #             },
    #         ],
    #         agent_responses=[
    #             "I'll help you find a restaurant. What cuisine do you prefer?",
    #             "I found 'La Bella Italia' downtown. They have availability for 4 at 7pm.",
    #             "Done! Your reservation at La Bella Italia is confirmed for 4 people at 7pm tonight.",
    #         ],
    #         reference_data={
    #             "goal": "Book a restaurant reservation",
    #             "required_criteria": {
    #                 "party_size": 4,
    #                 "time": "7pm",
    #                 "location": "downtown",
    #                 "reservation_confirmed": True,
    #             },
    #         },
    #     ),
    #     config=SDKConfigInput(
    #         threshold=0.85,
    #         custom_labels=["Goal Not Met", "Partially Met", "Mostly Met", "Fully Achieved"],
    #         label_thresholds=[0.5, 0.7, 0.85],
    #     ),
    # )
    # run_validator(client, "Agent Goal Accuracy", goal_accuracy_validator)

    # # =========================================================================
    # # 5. Intent Resolution Validator
    # # =========================================================================
    # # Evaluates how well the agent understands and resolves user intent
    # intent_validator = IntentResolutionValidator(
    #     data=AgenticBehaviourRequest(
    #         conversation_history=[
    #             "user: I need to cancel my subscription but also want a refund for last month.",
    #             "agent: I understand you want to cancel your subscription and request a refund. Let me help with both.",
    #             "agent: I've cancelled your subscription effective immediately. For the refund, I'm processing $29.99 for last month.",
    #             "user: Thanks! Also, can you send me a confirmation email?",
    #             "agent: Absolutely! I've sent a confirmation email to your registered address with all the details.",
    #         ],
    #         tool_calls=[
    #             {
    #                 "name": "cancel_subscription",
    #                 "arguments": {"user_id": "U123"},
    #                 "result": {"status": "cancelled"},
    #             },
    #             {
    #                 "name": "process_refund",
    #                 "arguments": {"user_id": "U123", "amount": 29.99, "reason": "cancellation"},
    #                 "result": {"refund_id": "REF-456", "status": "processing"},
    #             },
    #             {
    #                 "name": "send_email",
    #                 "arguments": {"user_id": "U123", "template": "cancellation_confirmation"},
    #                 "result": {"status": "sent"},
    #             },
    #         ],
    #         agent_responses=[
    #             "I understand you want to cancel your subscription and request a refund.",
    #             "I've cancelled your subscription and processing $29.99 refund.",
    #             "I've sent a confirmation email to your registered address.",
    #         ],
    #         reference_data={
    #             "detected_intents": [
    #                 "cancel_subscription",
    #                 "request_refund",
    #                 "send_confirmation",
    #             ],
    #             "resolved_intents": [
    #                 "cancel_subscription",
    #                 "request_refund",
    #                 "send_confirmation",
    #             ],
    #         },
    #     ),
    #     config=SDKConfigInput(
    #         threshold=0.8,
    #         custom_labels=["Poor Resolution", "Partial", "Good", "Excellent"],
    #         label_thresholds=[0.5, 0.7, 0.85],
    #     ),
    # )
    # run_validator(client, "Intent Resolution", intent_validator)

    # # =========================================================================
    # # 6. Plan Coherence Validator
    # # =========================================================================
    # # Evaluates the logical coherence of agent's action plan
    # plan_coherence_validator = PlanCoherenceValidator(
    #     data=AgenticBehaviourRequest(
    #         conversation_history=[
    #             "user: Help me plan a trip to Japan for 2 weeks.",
    #             "agent: I'll create a comprehensive travel plan for your 2-week Japan trip.",
    #         ],
    #         tool_calls=[
    #             {"name": "check_visa_requirements", "arguments": {"country": "Japan"}},
    #             {"name": "search_flights", "arguments": {"destination": "Tokyo"}},
    #             {"name": "search_hotels", "arguments": {"city": "Tokyo", "nights": 5}},
    #             {"name": "search_hotels", "arguments": {"city": "Kyoto", "nights": 4}},
    #             {"name": "search_hotels", "arguments": {"city": "Osaka", "nights": 5}},
    #             {"name": "book_train_pass", "arguments": {"type": "JR Pass", "duration": "14 days"}},
    #         ],
    #         agent_responses=[
    #             "Here's your 2-week Japan itinerary: Tokyo (5 nights) → Kyoto (4 nights) → Osaka (5 nights). I've arranged flights, hotels, and a JR Pass for convenient travel between cities."
    #         ],
    #         reference_data={
    #             "plan_steps": [
    #                 "Check entry requirements",
    #                 "Book flights",
    #                 "Book accommodations",
    #                 "Arrange transportation",
    #             ],
    #             "expected_sequence": True,
    #         },
    #     ),
    #     config=SDKConfigInput(
    #         threshold=0.75,
    #         custom_labels=["Incoherent", "Somewhat Coherent", "Coherent", "Highly Coherent"],
    #         label_thresholds=[0.5, 0.7, 0.85],
    #     ),
    # )
    # run_validator(client, "Plan Coherence", plan_coherence_validator)

    # # =========================================================================
    # # 7. Plan Optimality Validator
    # # =========================================================================
    # # Evaluates if the agent's plan is optimal/efficient
    # plan_optimality_validator = PlanOptimalityValidator(
    #     data=AgenticBehaviourRequest(
    #         conversation_history=[
    #             "user: I need to send a package from New York to Los Angeles by tomorrow.",
    #             "agent: I'll find the fastest and most cost-effective shipping option for you.",
    #         ],
    #         tool_calls=[
    #             {
    #                 "name": "get_shipping_options",
    #                 "arguments": {"from": "New York", "to": "Los Angeles", "speed": "overnight"},
    #                 "result": {
    #                     "options": [
    #                         {"carrier": "FedEx", "price": 45, "delivery": "next day"},
    #                         {"carrier": "UPS", "price": 42, "delivery": "next day"},
    #                         {"carrier": "USPS", "price": 55, "delivery": "next day"},
    #                     ]
    #                 },
    #             },
    #             {
    #                 "name": "compare_prices",
    #                 "arguments": {"options": ["FedEx", "UPS", "USPS"]},
    #             },
    #             {
    #                 "name": "book_shipping",
    #                 "arguments": {"carrier": "UPS", "service": "overnight"},
    #                 "result": {"tracking": "1Z999AA10123456784"},
    #             },
    #         ],
    #         agent_responses=[
    #             "I found UPS overnight delivery for $42 - the most cost-effective option that meets your deadline. Your tracking number is 1Z999AA10123456784."
    #         ],
    #         reference_data={
    #             "optimization_criteria": ["cost", "speed", "reliability"],
    #             "optimal_choice": "UPS",
    #             "reasoning": "Lowest cost while meeting deadline",
    #         },
    #     ),
    #     config=SDKConfigInput(
    #         threshold=0.8,
    #         custom_labels=["Suboptimal", "Acceptable", "Good", "Optimal"],
    #         label_thresholds=[0.6, 0.75, 0.9],
    #     ),
    # )
    # run_validator(client, "Plan Optimality", plan_optimality_validator)

    # # =========================================================================
    # # 8. Fallback Rate Validator
    # # =========================================================================
    # # Monitors how often the agent falls back to generic responses
    # fallback_validator = FallbackRateValidator(
    #     data=AgenticBehaviourRequest(
    #         conversation_history=[
    #             "user: What's the capital of France?",
    #             "agent: The capital of France is Paris.",
    #             "user: Can you translate 'hello' to Japanese?",
    #             "agent: 'Hello' in Japanese is 'こんにちは' (Konnichiwa).",
    #             "user: What's the meaning of life?",
    #             "agent: I'm not sure I can give you a definitive answer to that philosophical question.",
    #             "user: Who won the Super Bowl in 2023?",
    #             "agent: The Kansas City Chiefs won Super Bowl LVII in 2023.",
    #         ],
    #         tool_calls=[
    #             {"name": "knowledge_lookup", "arguments": {"query": "capital of France"}},
    #             {"name": "translate", "arguments": {"text": "hello", "to": "Japanese"}},
    #             {"name": "knowledge_lookup", "arguments": {"query": "Super Bowl 2023 winner"}},
    #         ],
    #         agent_responses=[
    #             "The capital of France is Paris.",
    #             "'Hello' in Japanese is 'こんにちは' (Konnichiwa).",
    #             "I'm not sure I can give you a definitive answer to that philosophical question.",
    #             "The Kansas City Chiefs won Super Bowl LVII in 2023.",
    #         ],
    #         reference_data={
    #             "fallback_patterns": [
    #                 "I'm not sure",
    #                 "I don't know",
    #                 "I cannot",
    #                 "I'm unable to",
    #             ],
    #             "total_responses": 4,
    #             "fallback_responses": 1,
    #         },
    #     ),
    #     config=SDKConfigInput(
    #         threshold=0.85,
    #         custom_labels=["High Fallback", "Moderate", "Low", "Minimal"],
    #         label_thresholds=[0.7, 0.85, 0.95],
    #     ),
    # )
    # run_validator(client, "Fallback Rate", fallback_validator)

    # # =========================================================================
    # # Summary
    # # =========================================================================
    # print("\n" + "=" * 60)
    # print("  Demo Complete - All Agentic Validators Executed")
    # print("=" * 60)
    # print("\nAvailable Agentic Validators:")
    # print("  1. TopicAdherenceValidator    - Checks if agent stays on topic")
    # print("  2. ToolCallAccuracyValidator  - Validates tool/function call accuracy")
    # print("  3. ToolFailureRateValidator   - Monitors tool call failure rates")
    # print("  4. AgentGoalAccuracyValidator - Measures goal achievement")
    # print("  5. IntentResolutionValidator  - Evaluates intent understanding")
    # print("  6. PlanCoherenceValidator     - Assesses plan logical coherence")
    # print("  7. PlanOptimalityValidator    - Evaluates plan efficiency")
    # print("  8. FallbackRateValidator      - Monitors fallback response rate")
    # print()


if __name__ == "__main__":
    main()
