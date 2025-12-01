# Disseqt SDK for Python

Python SDK for Disseqt validators via the Dataset API. Decorator-based dynamic registry. Enum-driven slugs. Normalized responses with a dynamic `others` bag.

## Features

- **Clean API**: Single `client.validate(request)` method for all validators
- **Type Safety**: Full typing support with Python 3.12+
- **Auto-Registration**: Decorator-based validator registration system
- **Normalized Responses**: Consistent response format with dynamic `others` bag
- **Domain-Specific Models**: Module-scoped request types for each validation domain
- **Enum-Driven**: No raw strings in public API, everything uses enums

## Installation

### From GitHub (Current Method)

```bash
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git
```

For specific versions:

```bash
# Specific branch
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git@main

# Specific tag (when available)
pip install git+https://github.com/DisseqtAI/disseqt-python-sdk.git@v0.1.0
```

### From PyPI (Coming Soon)

```bash
pip install disseqt-sdk
```

For detailed installation instructions including private repository access, virtual environments, and troubleshooting, see [INSTALL.md](INSTALL.md).

## Quick Start

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
- **ToxicityValidator**: Detects toxic content in input text
- **SecurityValidator**: Identifies security issues in input

### Output Validation  
- **FactualConsistencyValidator**: Checks factual accuracy of output

### RAG Grounding
- **ContextRelevanceValidator**: Validates context relevance in RAG systems

### Agentic Behavior
- **TopicAdherenceValidator**: Ensures agents stay on topic

### MCP Security
- **McpPromptInjectionValidator**: Detects prompt injection attempts

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
    base_url="https://dataset-api-eu.disseqt.ai",  # Default
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
git clone <repository>
cd disseqt-sdk-python
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
