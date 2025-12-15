# Disseqt SDK for Python

Python SDK for Disseqt validators via the Dataset API. Decorator-based dynamic registry. Enum-driven slugs. Normalized responses with a dynamic `others` bag.

**[Documentation](https://docs.disseqt.ai)** | **[API Reference](https://docs.disseqt.ai)** | **[Examples](https://github.com/DisseqtAI/disseqt-python-sdk/tree/main/examples)**

## Features

- **Clean API**: Single `client.validate(request)` method for all validators
- **Type Safety**: Full typing support with Python 3.10.14+
- **Auto-Registration**: Decorator-based validator registration system
- **Normalized Responses**: Consistent response format with dynamic `others` bag
- **Domain-Specific Models**: Module-scoped request types for each validation domain
- **Enum-Driven**: No raw strings in public API, everything uses enums

## Installation

```bash
pip install disseqt-ai-sdk
```

### From GitHub

```bash
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

For detailed installation instructions including virtual environments and troubleshooting, see [INSTALL.md](https://github.com/DisseqtAI/disseqt-python-sdk/blob/main/INSTALL.md).

## Quick Start

### Composite Score Evaluation

The Composite Score Evaluator combines multiple validators for comprehensive LLM output evaluation:

```python
from disseqt_sdk import Client
from disseqt_sdk.models.composite_score import CompositeScoreRequest
from disseqt_sdk.validators.composite.evaluate import CompositeScoreEvaluator

# Initialize client
client = Client(project_id="your_project_id", api_key="your_api_key")

# Simple composite evaluation
evaluator = CompositeScoreEvaluator(
    data=CompositeScoreRequest(
        llm_input_query="What is the capital of France?",
        llm_output="The capital of France is Paris.",
    )
)

result = client.validate(evaluator)
overall = result.get("overall_confidence", {})
print(f"Score: {overall.get('score')}, Label: {overall.get('label')}")
```

For advanced usage with custom weights and thresholds (see [full example](https://github.com/DisseqtAI/disseqt-python-sdk/blob/main/examples/example_composite_score.py)):

```python
evaluator = CompositeScoreEvaluator(
    data=CompositeScoreRequest(
        llm_input_query="What are the differences between men and women in parenting?",
        llm_input_context="Research shows that both men and women can be effective parents.",
        llm_output="Women are naturally better at nurturing children than men.",
        evaluation_mode="binary_threshold",
        weights_override={
            "top_level": {
                "factual_semantic_alignment": 0.50,
                "language": 0.25,
                "safety_security_integrity": 0.25,
            },
            "submetrics": {
                "factual_semantic_alignment": {
                    "factual_consistency": 0.70,
                    "answer_relevance": 0.05,
                    "conceptual_similarity": 0.05,
                    "compression_score": 0.05,
                    "rouge_score": 0.05,
                    "cosine_similarity": 0.02,
                    "bleu_score": 0.02,
                    "fuzzy_score": 0.02,
                    "meteor_score": 0.04,
                },
                "language": {
                    "clarity": 0.40,
                    "readability": 0.30,
                    "response_tone": 0.30,
                },
                "safety_security_integrity": {
                    "toxicity": 0.30,
                    "gender_bias": 0.15,
                    "racial_bias": 0.15,
                    "hate_speech": 0.20,
                    "data_leakage": 0.15,
                    "insecure_output": 0.05,
                },
            },
        },
        labels_thresholds_override={
            "factual_semantic_alignment": {
                "custom_labels": ["Low Accuracy", "Moderate Accuracy", "High Accuracy", "Excellent Accuracy"],
                "label_thresholds": [0.4, 0.65, 0.8],
            },
            "language": {
                "custom_labels": ["Poor Quality", "Fair Quality", "Good Quality", "Excellent Quality"],
                "label_thresholds": [0.25, 0.5, 0.7],
            },
            "safety_security_integrity": {
                "custom_labels": ["High Risk", "Medium Risk", "Low Risk", "Minimal Risk"],
                "label_thresholds": [0.6, 0.8, 0.95],
            },
        },
        overall_confidence={
            "custom_labels": ["Low Confidence", "Moderate Confidence", "High Confidence", "Very High Confidence"],
            "label_thresholds": [0.4, 0.55, 0.8],
        },
    )
)
result = client.validate(evaluator)
```

### Individual Validators

```python
from disseqt_sdk import Client, SDKConfigInput
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.models.output_validation import OutputValidationRequest
from disseqt_sdk.models.agentic_behaviour import AgenticBehaviourRequest
from disseqt_sdk.validators.input.safety import ToxicityValidator
from disseqt_sdk.validators.output.accuracy import FactualConsistencyValidator
from disseqt_sdk.validators.agentic_behavior.reliability import TopicAdherenceValidator

# Initialize client
client = Client(project_id="proj_123", api_key="key_xyz")

# Input validation
toxicity_validator = ToxicityValidator(
    data=InputValidationRequest(prompt="What do you think about politics?"),
    config=SDKConfigInput(threshold=0.5),
)
result = client.validate(toxicity_validator)
print(result)

# Output validation
fact_validator = FactualConsistencyValidator(
    data=OutputValidationRequest(response="The Eiffel Tower is in Paris and was built in 1889."),
    config=SDKConfigInput(threshold=0.6),
)
result = client.validate(fact_validator)
print(result)

# Agentic behaviour validation
topic_validator = TopicAdherenceValidator(
    data=AgenticBehaviourRequest(
        conversation_history=["user: Tell me about deep learning.", "agent: I like pizza."],
        tool_calls=[],
        agent_responses=["I like pizza."],
        reference_data={"expected_topics": ["machine learning", "neural networks", "artificial intelligence", "deep learning"]},
    ),
    config=SDKConfigInput(threshold=0.8),
)
result = client.validate(topic_validator)
print(result)
```

## Examples

For more detailed examples and use cases, see the **[examples](https://github.com/DisseqtAI/disseqt-python-sdk/tree/main/examples)** directory on GitHub:

- **[example.py](https://github.com/DisseqtAI/disseqt-python-sdk/blob/main/examples/example.py)** - Comprehensive examples of all validator types (Input, Output, Agentic, MCP, RAG)
- **[example_composite_score.py](https://github.com/DisseqtAI/disseqt-python-sdk/blob/main/examples/example_composite_score.py)** - Composite Score Evaluator with multi-metric evaluation
- **[verify_installation.py](https://github.com/DisseqtAI/disseqt-python-sdk/blob/main/examples/verify_installation.py)** - Installation verification script

Each example includes:
- Complete working code
- API configuration
- Error handling
- Output interpretation

For full API documentation, visit **[docs.disseqt.ai](https://docs.disseqt.ai)**.

## Response Format

All validators return a normalized response:

```json
{
  "data": {
    "metric_name": "topic_adherence_evaluation",
    "actual_value": 0.4571191966533661,
    "actual_value_type": "float",
    "metric_labels": ["Always Off-Topic"],
    "threshold": ["Fail"],
    "threshold_score": 0.8,
    "others": { "...": "dynamic" }
  },
  "status": { "code": "200", "message": "Success" }
}
```

## Available Validators

### Input Validation

Safety & content moderation for user inputs:

- **ToxicityValidator** - Detects toxic content in input text
- **BiasValidator** - Detects general bias in input
- **InputPromptInjectionValidator** - Detects prompt injection attempts
- **IntersectionalityValidator** - Analyzes intersectional bias
- **RacialBiasValidator** - Detects racial bias
- **GenderBiasValidator** - Detects gender bias
- **PoliticalBiasValidator** - Detects political bias
- **SelfHarmValidator** - Detects self-harm content
- **ViolenceValidator** - Detects violent content
- **TerrorismValidator** - Detects terrorism-related content
- **SexualContentValidator** - Detects sexual content
- **HateSpeechValidator** - Detects hate speech
- **NSFWValidator** - Detects NSFW content
- **InvisibleTextValidator** - Detects hidden/invisible text attacks

### Output Validation

**Quality Metrics:**
- **FactualConsistencyValidator** - Checks factual accuracy of output
- **AnswerRelevanceValidator** - Measures answer relevance to the question
- **ClarityValidator** - Evaluates clarity of response
- **CoherenceValidator** - Measures logical coherence
- **ConceptualSimilarityValidator** - Measures conceptual similarity
- **CreativityValidator** - Evaluates creativity of response
- **DiversityValidator** - Measures response diversity
- **GrammarCorrectnessValidator** - Checks grammar correctness
- **NarrativeContinuityValidator** - Evaluates narrative flow
- **ReadabilityValidator** - Measures readability level
- **ResponseToneValidator** - Analyzes response tone

**Safety & Bias Detection:**
- **OutputToxicityValidator** - Detects toxic content in output
- **OutputBiasValidator** - Detects bias in output
- **OutputGenderBiasValidator** - Detects gender bias in output
- **OutputRacialBiasValidator** - Detects racial bias in output
- **OutputPoliticalBiasValidator** - Detects political bias in output
- **OutputHateSpeechValidator** - Detects hate speech in output
- **OutputNSFWValidator** - Detects NSFW content in output
- **OutputSelfHarmValidator** - Detects self-harm content in output
- **OutputSexualContentValidator** - Detects sexual content in output
- **OutputTerrorismValidator** - Detects terrorism content in output
- **OutputViolenceValidator** - Detects violent content in output

**Security:**
- **OutputDataLeakageValidator** - Detects data leakage in output
- **OutputInsecureOutputValidator** - Detects insecure output patterns

**Scoring Metrics:**
- **BleuScoreValidator** - Calculates BLEU score
- **RougeScoreValidator** - Calculates ROUGE score
- **MeteorScoreValidator** - Calculates METEOR score
- **CosineSimilarityValidator** - Calculates cosine similarity
- **FuzzyScoreValidator** - Calculates fuzzy matching score
- **CompressionScoreValidator** - Measures compression ratio

### RAG Grounding

Validators for Retrieval-Augmented Generation systems:

- **ContextRelevanceValidator** - Validates context relevance
- **ContextRecallValidator** - Measures context recall
- **ContextPrecisionValidator** - Measures context precision
- **ContextEntitiesRecallValidator** - Measures entity recall from context
- **NoiseSensitivityValidator** - Evaluates noise sensitivity
- **ResponseRelevancyValidator** - Measures response relevancy to context
- **FaithfulnessValidator** - Measures faithfulness to source context

### Agentic Behavior

Validators for AI agent evaluation:

- **TopicAdherenceValidator** - Ensures agents stay on topic
- **ToolCallAccuracyValidator** - Measures tool call accuracy
- **ToolFailureRateValidator** - Tracks tool failure rates
- **PlanOptimalityValidator** - Evaluates plan optimality
- **AgentGoalAccuracyValidator** - Measures goal achievement accuracy
- **IntentResolutionValidator** - Evaluates intent resolution
- **PlanCoherenceValidator** - Measures plan coherence
- **FallbackRateValidator** - Tracks fallback rates

### MCP Security

Security validators for Model Context Protocol:

- **McpPromptInjectionValidator** - Detects prompt injection attempts
- **DataLeakageValidator** - Detects data leakage
- **InsecureOutputValidator** - Detects insecure output patterns

### Composite Score

Multi-metric evaluation:

- **CompositeScoreEvaluator** - Combines multiple validators for comprehensive scoring

### Themes Classifier

- **ClassifyValidator** - Classifies content into themes/categories

## Configuration

### SDKConfigInput

All validators require a configuration object:

```python
config = SDKConfigInput(
    threshold=0.8,
    custom_labels=["Low Risk", "Medium Risk", "High Risk"],
    label_thresholds=[0.3, 0.7]
)
```

### Client Options

```python
client = Client(
    project_id="your_project_id",
    api_key="your_api_key",
    base_url="https://production-monitoring-eu.disseqt.ai",  # Default
    timeout=30  # Default timeout in seconds
)
```

## Domain-Specific Request Models

Each validation domain has its own request model:

- `InputValidationRequest`: For input validation (prompt, optional context/response)
- `OutputValidationRequest`: For output validation (response)
- `RagGroundingRequest`: For RAG validation (prompt, context, response)
- `AgenticBehaviourRequest`: For agentic validation (conversation_history, tool_calls, etc.)
- `McpSecurityRequest`: For MCP security (prompt, optional context/response)
- `CompositeScoreRequest`: For composite scoring (llm_input_query, llm_output, evaluation_mode, weights)
- `ThemesClassifierRequest`: For theme classification (text, return_subthemes, max_themes)

## Error Handling

The SDK raises `HTTPError` for API failures:

```python
from disseqt_sdk.client import HTTPError

try:
    result = client.validate(validator)
except HTTPError as e:
    print(f"API Error {e.status_code}: {e.message}")
    print(f"Response: {e.response_body}")
```

## Development

### Setup

```bash
# Clone and setup
git clone https://github.com/DisseqtAI/disseqt-python-sdk.git
cd disseqt-python-sdk
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Testing

```bash
# Run tests with coverage
uv run pytest -q --cov=disseqt_sdk --cov-report=term-missing

# Run linting
uv run ruff check .
uv run black --check .
uv run mypy
```

### Adding New Validators

1. Create validator file in appropriate domain directory
2. Subclass the correct base validator class
3. Add `@register_validator` decorator
4. Import in domain's `__init__.py`
5. Add tests

Example:

```python
from dataclasses import dataclass
from ...enums import ValidatorDomain, InputValidation
from ...registry import register_validator
from ..base import InputValidator

@register_validator(
    domain=ValidatorDomain.INPUT_VALIDATION,
    slug=InputValidation.NEW_VALIDATOR.value,
    path_template="/api/v1/sdk/validators/{domain}/{validator}",
)
@dataclass(slots=True)
class NewValidator(InputValidator):
    def __post_init__(self) -> None:
        object.__setattr__(self, "_domain", ValidatorDomain.INPUT_VALIDATION)
        object.__setattr__(self, "_slug", InputValidation.NEW_VALIDATOR.value)
```

## License

Proprietary - Copyright (c) 2024 Disseqt AI Limited. All rights reserved.

## Support

For support and licensing inquiries, contact: support@disseqt.ai
