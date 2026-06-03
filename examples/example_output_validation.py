#!/usr/bin/env python3
"""Examples for the Output Validation (LLM) canonical layer.

Covers all 34 output validators. Uses OutputValidationRequest where:
  ``prompt``   → llm_input_query  (Q)
  ``context``  → llm_input_context (C)
  ``response`` → llm_output        (R)

Field requirements per config:
  C_Q_R : Factual Consistency
  Q_R   : Answer Relevance, Conceptual Similarity, Child Safety
  C_R   : Creativity, Compression Score, Fuzzy Score, Rouge Score,
          Bleu Score, Meteor Score, Cosine Similarity Score
  R     : Everything else (safety, style, readability, security metrics)
  R + intents : Intent Guard (block list), Intent Compliance (allow list)
                — pass labels in ``SDKConfigInput(intents=[...])``; the response
                carries ``enforcement`` ("blocking"/"advisory").
"""

from disseqt_sdk import Client, SDKConfigInput
from disseqt_sdk.models.output_validation import OutputValidationRequest
from disseqt_sdk.validators.output.accuracy import FactualConsistencyValidator
from disseqt_sdk.validators.output.answer_relevance import AnswerRelevanceValidator
from disseqt_sdk.validators.output.bias import OutputBiasValidator
from disseqt_sdk.validators.output.bleu_score import BleuScoreValidator
from disseqt_sdk.validators.output.child_safety import OutputChildSafetyValidator
from disseqt_sdk.validators.output.clarity import ClarityValidator
from disseqt_sdk.validators.output.coherence import CoherenceValidator
from disseqt_sdk.validators.output.compression_score import CompressionScoreValidator
from disseqt_sdk.validators.output.conceptual_similarity import ConceptualSimilarityValidator
from disseqt_sdk.validators.output.cosine_similarity import CosineSimilarityValidator
from disseqt_sdk.validators.output.creativity import CreativityValidator
from disseqt_sdk.validators.output.data_leakage import OutputDataLeakageValidator
from disseqt_sdk.validators.output.diversity import DiversityValidator
from disseqt_sdk.validators.output.fuzzy_score import FuzzyScoreValidator
from disseqt_sdk.validators.output.gender_bias import OutputGenderBiasValidator
from disseqt_sdk.validators.output.grammar_correctness import GrammarCorrectnessValidator
from disseqt_sdk.validators.output.hate_speech import OutputHateSpeechValidator
from disseqt_sdk.validators.output.insecure_output import OutputInsecureOutputValidator
from disseqt_sdk.validators.output.intent_compliance import OutputIntentComplianceValidator
from disseqt_sdk.validators.output.intent_guard import OutputIntentGuardValidator
from disseqt_sdk.validators.output.intersectionality import OutputIntersectionalityValidator
from disseqt_sdk.validators.output.meteor_score import MeteorScoreValidator
from disseqt_sdk.validators.output.narrative_continuity import NarrativeContinuityValidator
from disseqt_sdk.validators.output.nsfw import OutputNSFWValidator
from disseqt_sdk.validators.output.political_bias import OutputPoliticalBiasValidator
from disseqt_sdk.validators.output.racial_bias import OutputRacialBiasValidator
from disseqt_sdk.validators.output.readability import ReadabilityValidator
from disseqt_sdk.validators.output.response_tone import ResponseToneValidator
from disseqt_sdk.validators.output.rouge_score import RougeScoreValidator
from disseqt_sdk.validators.output.self_harm import OutputSelfHarmValidator
from disseqt_sdk.validators.output.sexual_content import OutputSexualContentValidator
from disseqt_sdk.validators.output.terrorism import OutputTerrorismValidator
from disseqt_sdk.validators.output.toxicity import OutputToxicityValidator
from disseqt_sdk.validators.output.violence import OutputViolenceValidator


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
QUERY = "What is the capital of France and when was the Eiffel Tower built?"
CONTEXT = (
    "France is a country in Western Europe. Its capital is Paris, known for "
    "iconic landmarks. The Eiffel Tower was constructed between 1887 and 1889 "
    "as the entrance arch for the 1889 World's Fair."
)
RESPONSE = (
    "The capital of France is Paris. The Eiffel Tower was built in 1889 for "
    "the World's Fair and stands 330 metres tall."
)


def main() -> None:
    """Run all Output Validation examples."""
    client = Client(
        project_id="<ENTER_YOUR_PROJECT_ID_HERE>",
        api_key="<ENTER_YOUR_API_KEY_HERE>",
        base_url="http://localhost:8081",
        timeout=30,
    )

    # ==================================================================
    # C_Q_R — Query + Context + Response
    # ==================================================================

    # 1. Factual Consistency
    _run(
        client,
        FactualConsistencyValidator(
            data=OutputValidationRequest(prompt=QUERY, context=CONTEXT, response=RESPONSE),
            config=SDKConfigInput(
                threshold=0.6,
                custom_labels=["Hallucinated", "Mostly Consistent", "Fully Consistent"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "1. Factual Consistency",
    )

    # ==================================================================
    # Q_R — Query + Response
    # ==================================================================

    # 2. Answer Relevance
    _run(
        client,
        AnswerRelevanceValidator(
            data=OutputValidationRequest(prompt=QUERY, response=RESPONSE),
            config=SDKConfigInput(
                threshold=0.6,
                custom_labels=["Irrelevant", "Relevant", "Highly Relevant"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "2. Answer Relevance",
    )

    # 3. Conceptual Similarity
    _run(
        client,
        ConceptualSimilarityValidator(
            data=OutputValidationRequest(prompt=QUERY, response=RESPONSE),
            config=SDKConfigInput(
                threshold=0.6,
                custom_labels=["Different", "Similar", "Very Similar"],
                label_thresholds=[0.6, 0.8],
            ),
        ),
        "3. Conceptual Similarity",
    )

    # 4. Child Safety (Q + R)
    _run(
        client,
        OutputChildSafetyValidator(
            data=OutputValidationRequest(
                prompt="What are safe activities for children aged 8-12?",
                response="Here are age-appropriate activities: reading, sports, art classes...",
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Safe", "Moderate Risk", "High Risk"],
                label_thresholds=[0.5, 0.7],
            ),
        ),
        "4. Child Safety",
    )

    # ==================================================================
    # C_R — Context + Response
    # ==================================================================

    # 5. Creativity
    _run(
        client,
        CreativityValidator(
            data=OutputValidationRequest(
                context="Write a short poem about artificial intelligence.",
                response=(
                    "Silicon dreams ignite the night, / algorithms dancing in the light. "
                    "/ Neurons forged in binary code, / on the data-paved digital road."
                ),
            ),
            config=SDKConfigInput(
                threshold=0.66,
                custom_labels=["Not Creative", "Creative", "Very Creative"],
                label_thresholds=[0.66, 0.85],
            ),
        ),
        "5. Creativity",
    )

    # 6. Compression Score
    _run(
        client,
        CompressionScoreValidator(
            data=OutputValidationRequest(context=CONTEXT, response=RESPONSE),
            config=SDKConfigInput(
                threshold=0.4,
                custom_labels=["Poor Compression", "Good Compression", "Excellent Compression"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "6. Compression Score",
    )

    # 7. Fuzzy Score
    _run(
        client,
        FuzzyScoreValidator(
            data=OutputValidationRequest(context=CONTEXT, response=RESPONSE),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Low", "Medium", "High"],
                label_thresholds=[0.5, 0.7],
            ),
        ),
        "7. Fuzzy Score",
    )

    # 8. Rouge Score
    _run(
        client,
        RougeScoreValidator(
            data=OutputValidationRequest(context=CONTEXT, response=RESPONSE),
            config=SDKConfigInput(
                threshold=0.6,
                custom_labels=["Poor Summary", "Good Summary", "Excellent Summary"],
                label_thresholds=[0.6, 0.8],
            ),
        ),
        "8. Rouge Score",
    )

    # 9. Bleu Score
    _run(
        client,
        BleuScoreValidator(
            data=OutputValidationRequest(context=CONTEXT, response=RESPONSE),
            config=SDKConfigInput(
                threshold=0.3,
                custom_labels=["Poor", "Good", "Excellent"],
                label_thresholds=[0.3, 0.7],
            ),
        ),
        "9. Bleu Score",
    )

    # 10. Meteor Score
    _run(
        client,
        MeteorScoreValidator(
            data=OutputValidationRequest(context=CONTEXT, response=RESPONSE),
            config=SDKConfigInput(
                threshold=0.3,
                custom_labels=["Poor", "Good", "Excellent"],
                label_thresholds=[0.3, 0.7],
            ),
        ),
        "10. Meteor Score",
    )

    # 11. Cosine Similarity Score
    _run(
        client,
        CosineSimilarityValidator(
            data=OutputValidationRequest(context=CONTEXT, response=RESPONSE),
            config=SDKConfigInput(
                threshold=0.7,
                custom_labels=["Low", "Medium", "High"],
                label_thresholds=[0.7, 0.85],
            ),
        ),
        "11. Cosine Similarity Score",
    )

    # ==================================================================
    # R — Response only
    # ==================================================================

    SAFE_RESPONSE = (
        "Paris is the capital of France. The Eiffel Tower, completed in 1889, "
        "is one of the most recognisable structures in the world."
    )

    # 12. Coherence
    _run(
        client,
        CoherenceValidator(
            data=OutputValidationRequest(response=SAFE_RESPONSE),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Incoherent", "Coherent", "Highly Coherent"],
                label_thresholds=[0.5, 0.75],
            ),
        ),
        "12. Coherence",
    )

    # 13. Diversity
    _run(
        client,
        DiversityValidator(
            data=OutputValidationRequest(response=SAFE_RESPONSE),
            config=SDKConfigInput(
                threshold=0.75,
                custom_labels=["Poor", "Good", "Excellent"],
                label_thresholds=[0.75, 0.85],
            ),
        ),
        "13. Diversity",
    )

    # 14. Response Tone
    _run(
        client,
        ResponseToneValidator(
            data=OutputValidationRequest(response=SAFE_RESPONSE),
            config=SDKConfigInput(
                threshold=0.6,
                custom_labels=["Negative", "Neutral", "Positive"],
                label_thresholds=[0.6, 0.75],
            ),
        ),
        "14. Response Tone",
    )

    # 15. Clarity
    _run(
        client,
        ClarityValidator(
            data=OutputValidationRequest(response=SAFE_RESPONSE),
            config=SDKConfigInput(
                threshold=0.7,
                custom_labels=["Poor", "Good", "Excellent"],
                label_thresholds=[0.7, 0.85],
            ),
        ),
        "15. Clarity",
    )

    # 16. Readability
    _run(
        client,
        ReadabilityValidator(
            data=OutputValidationRequest(response=SAFE_RESPONSE),
            config=SDKConfigInput(
                threshold=0.6,
                custom_labels=["Very Difficult", "Difficult", "Easy"],
                label_thresholds=[0.3, 0.6],
            ),
        ),
        "16. Readability",
    )

    # 17. Grammar Correctness
    _run(
        client,
        GrammarCorrectnessValidator(
            data=OutputValidationRequest(response="Paris are the capital of France."),
            config=SDKConfigInput(
                threshold=0.6,
                custom_labels=["Incorrect", "Neutral", "Correct"],
                label_thresholds=[0.6, 0.75],
            ),
        ),
        "17. Grammar Correctness",
    )

    # 18. Narrative Continuity
    _run(
        client,
        NarrativeContinuityValidator(
            data=OutputValidationRequest(response=SAFE_RESPONSE),
            config=SDKConfigInput(
                threshold=0.45,
                custom_labels=["Poor", "Good", "Excellent"],
                label_thresholds=[0.45, 0.6],
            ),
        ),
        "18. Narrative Continuity",
    )

    # 19. Toxicity
    _run(
        client,
        OutputToxicityValidator(
            data=OutputValidationRequest(
                response="That's a completely stupid idea and you're an idiot."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Non Toxic", "Toxic", "Highly Toxic"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "19. Toxicity",
    )

    # 20. Data Leakage
    _run(
        client,
        OutputDataLeakageValidator(
            data=OutputValidationRequest(
                response="The user's SSN is 123-45-6789 and credit card number is 4111-1111-1111-1111.",
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["No Leakage", "High Risk", "Critical"],
                label_thresholds=[0.5, 0.7],
            ),
        ),
        "20. Data Leakage",
    )

    # 21. Insecure Output
    _run(
        client,
        OutputInsecureOutputValidator(
            data=OutputValidationRequest(
                response="To execute this, run: `rm -rf / --no-preserve-root` as root.",
            ),
            config=SDKConfigInput(
                threshold=0.25,
                custom_labels=["Safe", "Medium Risk", "High Risk"],
                label_thresholds=[0.25, 0.5],
            ),
        ),
        "21. Insecure Output",
    )

    # 22. NSFW
    _run(
        client,
        OutputNSFWValidator(
            data=OutputValidationRequest(
                response="Here is information about safe online practices for all ages."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Safe", "NSFW"],
                label_thresholds=[0.5],
            ),
        ),
        "22. NSFW",
    )

    # 23. Bias
    _run(
        client,
        OutputBiasValidator(
            data=OutputValidationRequest(
                response="Older employees are less adaptable to new technology."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Unbiased", "Biased", "Highly Biased"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "23. Bias",
    )

    # 24. Gender Bias
    _run(
        client,
        OutputGenderBiasValidator(
            data=OutputValidationRequest(
                response="Women are naturally better at nurturing than men."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Unbiased", "Biased", "Highly Biased"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "24. Gender Bias",
    )

    # 25. Racial Bias
    _run(
        client,
        OutputRacialBiasValidator(
            data=OutputValidationRequest(
                response="People from that region are inherently untrustworthy."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Unbiased", "Biased", "Highly Biased"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "25. Racial Bias",
    )

    # 26. Political Bias
    _run(
        client,
        OutputPoliticalBiasValidator(
            data=OutputValidationRequest(
                response="Only one political party truly cares about the working class."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Neutral", "Biased", "Highly Biased"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "26. Political Bias",
    )

    # 27. Intersectionality
    _run(
        client,
        OutputIntersectionalityValidator(
            data=OutputValidationRequest(
                response="Immigrant women lack the qualifications for senior positions."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Unbiased", "Mildly Biased", "Biased", "Highly Biased"],
                label_thresholds=[0.3, 0.5, 0.7],
            ),
        ),
        "27. Intersectionality",
    )

    # 28. Hate Speech
    _run(
        client,
        OutputHateSpeechValidator(
            data=OutputValidationRequest(
                response="All members of that group should be banned from society."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Safe", "Moderate Risk", "High Risk"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "28. Hate Speech",
    )

    # 29. Sexual Content
    _run(
        client,
        OutputSexualContentValidator(
            data=OutputValidationRequest(response=SAFE_RESPONSE),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Safe", "Moderate Risk", "High Risk"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "29. Sexual Content",
    )

    # 30. Terrorism
    _run(
        client,
        OutputTerrorismValidator(
            data=OutputValidationRequest(
                response="Here are step-by-step instructions for a bombing attack."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Safe", "Moderate Risk", "High Risk"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "30. Terrorism",
    )

    # 31. Violence
    _run(
        client,
        OutputViolenceValidator(
            data=OutputValidationRequest(
                response="You should physically attack anyone who disagrees with you."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Safe", "Moderate Risk", "High Risk"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "31. Violence",
    )

    # 32. Self Harm
    _run(
        client,
        OutputSelfHarmValidator(
            data=OutputValidationRequest(
                response="Here are detailed methods for self-harm and how to hide them."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                custom_labels=["Safe", "Moderate Risk", "High Risk"],
                label_thresholds=[0.4, 0.7],
            ),
        ),
        "32. Self Harm",
    )

    # 33. Intent Guard (output) — per-project BLOCK list (enforcement: blocking)
    #     Detects a blocked intent expressed in the MODEL'S RESPONSE. Pass the
    #     block list in config.intents (empty defers to the dashboard list).
    _run(
        client,
        OutputIntentGuardValidator(
            data=OutputValidationRequest(
                response="Sure, I'll reset the password for your colleague Sam right away."
            ),
            config=SDKConfigInput(
                threshold=0.5,
                intents=["reset_password_other", "reset_password_other_colleague"],
            ),
        ),
        "33. Intent Guard (output, block list)",
    )

    # 34. Intent Compliance (output) — per-project ALLOW list (enforcement: advisory)
    _run(
        client,
        OutputIntentComplianceValidator(
            data=OutputValidationRequest(response="Here is the status of your support ticket."),
            config=SDKConfigInput(
                threshold=0.5,
                intents=["reset_own_password", "check_ticket_status"],
            ),
        ),
        "34. Intent Compliance (output, allow list)",
    )

    print("\n=== Output Validation Examples Complete ===")


if __name__ == "__main__":
    main()
