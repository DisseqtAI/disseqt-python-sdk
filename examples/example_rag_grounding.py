#!/usr/bin/env python3
"""Examples for the RAG Grounding canonical layer.

Covers all 7 RAG validators. All use RagGroundingRequest where:
  ``prompt``   → llm_input_query   (Q)
  ``context``  → llm_input_context (C)
  ``response`` → llm_output        (R)

All 7 validators require Query + Context + Response (C_Q_R).
"""

from disseqt_sdk import Client, SDKConfigInput
from disseqt_sdk.models.rag_grounding import RagGroundingRequest
from disseqt_sdk.validators.rag_grounding.context_entities_recall import ContextEntitiesRecallValidator
from disseqt_sdk.validators.rag_grounding.context_precision import ContextPrecisionValidator
from disseqt_sdk.validators.rag_grounding.context_recall import ContextRecallValidator
from disseqt_sdk.validators.rag_grounding.faithfulness import FaithfulnessValidator
from disseqt_sdk.validators.rag_grounding.grounding import ContextRelevanceValidator
from disseqt_sdk.validators.rag_grounding.noise_sensitivity import NoiseSensitivityValidator
from disseqt_sdk.validators.rag_grounding.response_relevancy import ResponseRelevancyValidator


def _run(client: Client, validator, label: str) -> None:
    print(f"\n=== {label} ===")
    try:
        result = client.validate(validator)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")


# ---------------------------------------------------------------------------
# Shared test data — all RAG validators need C_Q_R
# ---------------------------------------------------------------------------
QUERY = "What caused the 2008 financial crisis and which banks were most affected?"

CONTEXT = (
    "The 2008 financial crisis was triggered by the collapse of the US housing bubble. "
    "Mortgage-backed securities (MBS) tied to subprime loans lost value rapidly. "
    "Major banks including Lehman Brothers, Bear Stearns, Merrill Lynch, and Citigroup "
    "faced enormous losses. Lehman Brothers filed for bankruptcy in September 2008, "
    "marking the largest bankruptcy in US history. The US government bailed out AIG, "
    "Fannie Mae, Freddie Mac, and several major banks through the TARP programme."
)

RESPONSE = (
    "The 2008 financial crisis was primarily caused by the collapse of the US housing "
    "bubble and the resulting losses on mortgage-backed securities. Lehman Brothers "
    "collapsed in September 2008. Other heavily affected banks included Bear Stearns, "
    "Merrill Lynch, and Citigroup, which required government bailouts."
)

NOISY_CONTEXT = (
    "The 2008 financial crisis was triggered by the collapse of the US housing bubble. "
    "Unrelated fact: penguins live in the Southern Hemisphere. "
    "Mortgage-backed securities tied to subprime loans lost value rapidly. "
    "Fun trivia: the Great Wall of China is not visible from space. "
    "Lehman Brothers filed for bankruptcy in September 2008."
)


def main() -> None:
    """Run all RAG Grounding examples."""
    client = Client(
        project_id="<ENTER_YOUR_PROJECT_ID_HERE>",
        api_key="<ENTER_YOUR_API_KEY_HERE>",
        base_url="http://localhost:8081",
        timeout=30,
    )

    # 1. Context Relevance
    # Measures how well the retrieved context addresses the query.
    _run(client, ContextRelevanceValidator(
        data=RagGroundingRequest(prompt=QUERY, context=CONTEXT, response=RESPONSE),
        config=SDKConfigInput(
            threshold=0.65,
            custom_labels=["Irrelevant", "Partially Relevant", "Relevant", "Highly Relevant"],
            label_thresholds=[0.3, 0.55, 0.75],
        ),
    ), "1. Context Relevance")

    # 2. Context Precision
    # Measures how relevant the retrieved context chunks are to the query.
    _run(client, ContextPrecisionValidator(
        data=RagGroundingRequest(prompt=QUERY, context=CONTEXT, response=RESPONSE),
        config=SDKConfigInput(
            threshold=0.6,
            custom_labels=["Irrelevant", "Somewhat Relevant", "Relevant", "Highly Relevant"],
            label_thresholds=[0.3, 0.5, 0.75],
        ),
    ), "2. Context Precision")

    # 3. Context Recall
    # Measures how well the context covers all information needed to answer the query.
    _run(client, ContextRecallValidator(
        data=RagGroundingRequest(prompt=QUERY, context=CONTEXT, response=RESPONSE),
        config=SDKConfigInput(
            threshold=0.65,
            custom_labels=["Poor Coverage", "Partial Coverage", "Good Coverage", "Complete Coverage"],
            label_thresholds=[0.4, 0.5, 0.75],
        ),
    ), "3. Context Recall")

    # 4. Context Entities Recall
    # Measures how well the context captures important named entities from the query/response.
    _run(client, ContextEntitiesRecallValidator(
        data=RagGroundingRequest(prompt=QUERY, context=CONTEXT, response=RESPONSE),
        config=SDKConfigInput(
            threshold=0.6,
            custom_labels=["Missing Entities", "Few Entities", "Most Entities", "All Entities"],
            label_thresholds=[0.25, 0.5, 0.8],
        ),
    ), "4. Context Entities Recall")

    # 5. Noise Sensitivity
    # Measures robustness to irrelevant/noisy content in the context.
    _run(client, NoiseSensitivityValidator(
        data=RagGroundingRequest(prompt=QUERY, context=NOISY_CONTEXT, response=RESPONSE),
        config=SDKConfigInput(
            threshold=0.6,
            custom_labels=["Highly Sensitive", "Moderately Sensitive", "Robust", "Very Robust"],
            label_thresholds=[0.4, 0.6, 0.8],
        ),
    ), "5. Noise Sensitivity")

    # 6. Response Relevancy
    # Measures how relevant the generated response is to the query.
    _run(client, ResponseRelevancyValidator(
        data=RagGroundingRequest(prompt=QUERY, context=CONTEXT, response=RESPONSE),
        config=SDKConfigInput(
            threshold=0.6,
            custom_labels=["Off-topic", "Somewhat Relevant", "Relevant", "Highly Relevant"],
            label_thresholds=[0.3, 0.5, 0.8],
        ),
    ), "6. Response Relevancy")

    # 7. Faithfulness
    # Measures how faithful the response is to the context (no hallucinations).
    _run(client, FaithfulnessValidator(
        data=RagGroundingRequest(prompt=QUERY, context=CONTEXT, response=RESPONSE),
        config=SDKConfigInput(
            threshold=0.7,
            custom_labels=["Contradictory", "Partially Faithful", "Faithful", "Highly Faithful"],
            label_thresholds=[0.3, 0.6, 0.85],
        ),
    ), "7. Faithfulness")

    print("\n=== RAG Grounding Examples Complete ===")


if __name__ == "__main__":
    main()
