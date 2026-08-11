# Disseqt SDK for Python

Python SDK for Disseqt AI observability and validation. This package includes two SDKs:

1. **Validation SDK** (`disseqt_sdk`) - Validate LLM inputs, outputs, RAG, agentic behavior, and MCP security
2. **Agentic SDK** (`disseqt_agentic_sdk`) - Trace and monitor agentic AI workflows with OpenTelemetry-compatible spans

**[Documentation](https://docs.disseqt.ai)** | **[API Reference](https://docs.disseqt.ai)** | **[Examples](https://github.com/DisseqtAI/disseqt-python-sdk/tree/main/examples)**

## Features

### Validation SDK
- **Clean API**: Single `client.validate(request)` method for all validators
- **Type Safety**: Full typing support with Python 3.10.14+
- **Auto-Registration**: Decorator-based validator registration system
- **Normalized Responses**: Consistent response format with dynamic `others` bag
- **Domain-Specific Models**: Module-scoped request types for each validation domain
- **Enum-Driven**: No raw strings in public API, everything uses enums

### Agentic SDK
- **OpenTelemetry Compatible**: Standard span-based tracing
- **Multiple Span Kinds**: MODEL_EXEC, TOOL_EXEC, AGENT_EXEC, RAG_EXEC, MCP_EXEC, and custom kinds
- **Automatic Batching**: Efficient span batching and flushing
- **Context Management**: Thread-local context for nested traces
- **Helper Functions**: Decorators and utilities for easy integration

## Installation

```bash
pip install disseqt-ai-sdk
```

### From GitHub

```bash
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

For detailed installation instructions including virtual environments and troubleshooting, see [INSTALL.md](https://github.com/DisseqtAI/disseqt-python-sdk/blob/main/INSTALL.md).

## When to Use Which SDK?

### Use Agentic SDK (`disseqt_agentic_sdk`) when:
- ✅ You want to **trace and monitor** your agentic AI workflows
- ✅ You need **observability** for LLM calls, tool calls, and agent actions
- ✅ You want to track **span hierarchies** (parent-child relationships)
- ✅ You need **OpenTelemetry-compatible** tracing
- ✅ You want to **visualize workflows** in the Disseqt dashboard
- ✅ You're building **RAG systems** and need `RAG_EXEC` spans
- ✅ You're using **MCP (Model Context Protocol)** and need `MCP_EXEC` spans

### Use Validation SDK (`disseqt_sdk`) when:
- ✅ You want to **validate** LLM inputs/outputs for safety and quality
- ✅ You need to check for **toxicity, bias, prompt injection**, etc.
- ✅ You want to evaluate **RAG grounding** (context relevance, faithfulness)
- ✅ You need to assess **agentic behavior** (goal accuracy, tool call accuracy)
- ✅ You want **MCP security** validation (data leakage, insecure output)
- ✅ You need **composite scoring** across multiple metrics

## Quick Start

### Agentic SDK - Tracing Workflows

Trace your agentic AI workflows for observability and monitoring:

```python
from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.api.helpers import trace_llm_call, trace_tool_call

# Initialize client
client = DisseqtAgenticClient(
    api_key="your-api-key",
    project_id="proj_456",
    service_name="my-service",
    endpoint="http://localhost:8080/v1/traces"
)

# Create a trace with multiple spans
with start_trace(client, name="agent_workflow") as trace:
    # Agent execution span
    with trace.start_span("agent_execution", SpanKind.AGENT_EXEC) as agent_span:
        agent_span.set_agent_info("my_agent", "agent_001")
        
        # LLM call span
        with trace.start_span("llm_call", SpanKind.MODEL_EXEC) as llm_span:
            llm_span.set_model_info("gpt-4", "openai")
            llm_span.set_messages(
                input_messages=[{"role": "user", "content": "Hello"}],
                output_messages=[{"role": "assistant", "content": "Hi there!"}]
            )
            llm_span.set_token_usage(input_tokens=10, output_tokens=5)
        
        # Tool call span
        with trace.start_span("api_call", SpanKind.TOOL_EXEC) as tool_span:
            tool_span.set_tool_info("get_weather", "call_001")
            tool_span.set_attribute("tool.input.city", "Paris")

# Trace automatically sent when exiting the 'with' block
client.shutdown()
```

#### RAG Workflow Tracing

For RAG (Retrieval Augmented Generation) workflows, use `RAG_EXEC` span kind:

```python
with start_trace(client, name="rag_workflow") as trace:
    with trace.start_span("rag_retrieval", SpanKind.RAG_EXEC) as rag_span:
        # Set context and messages for RAG validation
        rag_span.set_attribute("agentic.input.context", "Retrieved context...")
        rag_span.set_messages(
            input_messages=[{"role": "user", "content": "What is AI?"}],
            output_messages=[{"role": "assistant", "content": "AI is..."}]
        )
```

#### MCP Workflow Tracing

For MCP (Model Context Protocol) workflows, use `MCP_EXEC` span kind:

```python
with start_trace(client, name="mcp_workflow") as trace:
    with trace.start_span("mcp_execution", SpanKind.MCP_EXEC) as mcp_span:
        mcp_span.set_messages(
            input_messages=[{"role": "user", "content": "Query"}],
            output_messages=[{"role": "assistant", "content": "Response"}]
        )
        mcp_span.set_attribute("mcp.protocol.version", "1.0")
```

#### Custom Span Kinds

You can use custom span kinds by passing any string:

```python
# Custom span kind
with trace.start_span("data_processing", "DATA_PROCESSING") as span:
    span.set_attribute("processing.type", "batch")

# Or with decorator
from disseqt_agentic_sdk.api.helpers import trace_function

@trace_function(client, kind="CUSTOM_OPERATION")
def my_function():
    return "result"
```

#### Available Span Kinds

| Span Kind | Description | Use Case |
|-----------|-------------|----------|
| `MODEL_EXEC` | LLM model execution | GPT-4, Claude, Gemini calls |
| `TOOL_EXEC` | Tool/function execution | API calls, calculator, database queries |
| `AGENT_EXEC` | Agent workflow execution | Main agent orchestration |
| `RAG_EXEC` | RAG execution | Retrieval + generation workflows (required for RAG validations) |
| `MCP_EXEC` | MCP protocol execution | Model Context Protocol interactions (required for MCP validations) |
| `INTERNAL` | Internal operations | Internal processing, data transformation |
| `CLIENT` | Client-side operations | Standard OTLP client span |
| `SERVER` | Server-side operations | Standard OTLP server span |
| Custom strings | Any custom category | `"DATA_PROCESSING"`, `"CUSTOM_OPERATION"`, etc. |

For more Agentic SDK examples, see the **[agentic examples](https://github.com/DisseqtAI/disseqt-python-sdk/tree/main/examples/agentic)** directory.

### Validation SDK - Composite Score Evaluation

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

### LLM-as-a-Judge (bring your own LLM)

Any paired validator can be graded by a certified LLM judge running on **your
own** LLM Integration (your provider account, your key) instead of the classic
ML model. Set `llm_as_a_judge=True`:

```python
config = SDKConfigInput(
    threshold=0.5,
    llm_as_a_judge=True,
    # Which integration judges the run. Copy this id from
    # Dashboard -> AI Inventory -> LLM Integrations (ID column).
    # Omit `judge` entirely to use the project's default judge integration.
    judge={"custom_llm_id": "25dc0684-a394-4389-ad59-90b27138badf"},
)

result = client.validate(
    ToxicityValidator(data=InputValidationRequest(prompt="hello"), config=config)
)

# The verdict carries receipts — trust these, not assumptions:
others = result["result"]["data"]["others"]
others["model"]           # which LLM actually judged (e.g. "gpt-5")
others["rubric_version"]  # the certified rubric that produced the score
others["reasoning"]       # the judge's written justification
others.get("scoring_path")     # quality judges: "logprob_weighted" | "rating_fallback"
result.get("origin_validator") # set when the judge reroute renamed the run
                               # (you asked for "toxicity", it ran "llm-judge-toxicity")
```

Notes:

- Validators **without** a judge pairing fall back gracefully to the ML path — no error.
- `judge["criteria"]` shapes **quality** judges only; certified **safety** judges run
  their frozen rubric verbatim (the response stamps `criteria_ignored` if you try).
- Judge inference runs on your provider account, so a rejected key or exhausted
  quota surfaces as an HTTP error from your provider's side — check your
  provider dashboard, not just the SDK call.

## Examples

### Agentic SDK Examples

For Agentic SDK (tracing/observability) examples, see the **[agentic examples](https://github.com/DisseqtAI/disseqt-python-sdk/tree/main/examples/agentic)** directory:

- **[example.py](https://github.com/DisseqtAI/disseqt-python-sdk/blob/main/examples/agentic/example.py)** - Complete example with multiple spans using context managers
- **[example_without_with.py](https://github.com/DisseqtAI/disseqt-python-sdk/blob/main/examples/agentic/example_without_with.py)** - Manual span management without `with` statements
- **[ai_consultant_agent.py](https://github.com/DisseqtAI/disseqt-python-sdk/blob/main/examples/agentic/ai_consultant_agent.py)** - Full AI agent integration example

### Validation SDK Examples

For Validation SDK examples, see the **[examples](https://github.com/DisseqtAI/disseqt-python-sdk/tree/main/examples)** directory:

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
- **ChildSafetyValidator** - Detects child-safety risks in input

Per-project intent guardrails (configurable block/allow lists):

- **IntentGuardValidator** - Blocks disallowed intents (block list; response `enforcement` "blocking")
- **IntentComplianceValidator** - Flags intents outside the allow list (`enforcement` "advisory")

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
- **OutputIntersectionalityValidator** - Detects intersectional bias in output
- **OutputChildSafetyValidator** - Detects child-safety risks in output

**Security:**
- **OutputDataLeakageValidator** - Detects data leakage in output
- **OutputInsecureOutputValidator** - Detects insecure output patterns

**Intent Guardrails:**
- **OutputIntentGuardValidator** - Blocks disallowed intents in the model's output (block list; `enforcement` "blocking")
- **OutputIntentComplianceValidator** - Flags output intents outside the allow list (`enforcement` "advisory")

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
    base_url="https://api.disseqt.ai/realtime-validations",  # Default
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

## Logging

The SDK ships a built-in structured logger (`disseqt_logging`) — **no extra
dependencies, nothing to install**. Both the validation and agentic SDKs emit a
single consistent log schema: JSON (or human console), automatic
`service` / `env` / `host` fields, PII/credential redaction, and a privacy-safe
payload digest.

**Silent by default.** The SDK produces **no log output** until you opt in via
`configure_logging(...)`, `set_log_level(...)`, or the `DISSEQT_LOG_LEVEL`
environment variable — so upgrading never adds unsolicited output. Once enabled,
`Client.validate()` logs one structured line per call:

```python
import disseqt_sdk
from disseqt_sdk import Client, SDKConfigInput
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.validators.input.safety import ToxicityValidator

disseqt_sdk.configure_logging(level="info")   # or set DISSEQT_LOG_LEVEL

client = Client(project_id="...", api_key="...")
client.validate(ToxicityValidator(
    data=InputValidationRequest(prompt="..."),
    config=SDKConfigInput(threshold=0.5),
))
# {"event": "validation.response", "domain": "input-validation",
#  "slug": "toxicity", "status": 200, "latency_ms": 142.3,
#  "service": "disseqt-ai-sdk", "env": "production", ...}
```

Safety guarantees baked in:

- Request payloads (which may contain user prompts) are **never logged
  verbatim** — only a content-free `digest` (`len=… sha256=…`).
- Auth headers, `api_key`, and `project_id` are never logged; as defense in
  depth, any field whose name looks sensitive (`*token*`, `*secret*`,
  `api_key`, `authorization`, `project_id`, …) and any email / JWT / card /
  phone / long-token shape in a string value is redacted.

### Configuration

Use `disseqt_sdk.configure_logging(...)` / `set_log_level(...)`, or the
environment:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DISSEQT_LOG_LEVEL` | _(unset → silent)_ | `debug` / `info` / `warn` / `error`; setting it enables output |
| `DISSEQT_LOG_FORMAT` | `auto` | `json`, `console`, or `auto` (console on a TTY) |
| `DISSEQT_ENV` | `production` | Value bound to the `env` field |
| `DISSEQT_LOG_SERVICE` | `disseqt-ai-sdk` | Value bound to the `service` field |
| `DISSEQT_LOG_REDACT` | `1` | Set to `0` to disable redaction |
| `DISSEQT_LOG_FILE` | — | When set, also tee to a size-rotated file |

The agentic SDK keeps its existing `get_logger()` / `set_log_level()` surface;
both now route through the same shared logger. Call `disseqt_logging.disable()`
to silence output again after enabling it.

**See [docs/logging.md](docs/logging.md)** for the full guide — using the logger
in your own code, structured fields, `bind()`, the error envelope, redaction
details, file output, and the complete API reference.

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
