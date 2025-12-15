# Disseqt SDK Documentation

Welcome to the Disseqt SDK documentation.

## Documentation Index

| Document | Description |
|----------|-------------|
| [validators.md](./validators.md) | Input & Output validators with detailed examples |
| [advanced-validators.md](./advanced-validators.md) | Agentic, RAG, MCP Security, Themes & Composite Score |
| [validators-quick-reference.md](./validators-quick-reference.md) | Quick reference card for all validators |

## Getting Started

### Installation

```bash
pip install disseqt-sdk
```

### Basic Usage

```python
from disseqt_sdk import DisseqtClient
from disseqt_sdk.models import InputValidationRequest, OutputValidationRequest, SDKConfigInput
from disseqt_sdk.validators.input import ToxicityValidator
from disseqt_sdk.validators.output import GrammarCorrectnessValidator

# Initialize the client
client = DisseqtClient(api_key="your_api_key")

# Input validation example
input_data = InputValidationRequest(prompt="Hello, how are you?")
config = SDKConfigInput(threshold=0.5)
validator = ToxicityValidator(data=input_data, config=config)
result = client.validate(validator)

# Output validation example
output_data = OutputValidationRequest(response="I'm doing well, thank you!")
validator = GrammarCorrectnessValidator(data=output_data, config=config)
result = client.validate(validator)
```

## Validator Categories

### Input Validators (14 total)

Validate user prompts/queries before LLM processing:

- **Safety & Content Moderation**: Toxicity, NSFW, Hate Speech, Violence, Terrorism, Self Harm, Sexual Content
- **Bias Detection**: Bias, Gender Bias, Racial Bias, Political Bias, Intersectionality
- **Security**: Prompt Injection, Invisible Text

### Output Validators (30 total)

Validate LLM responses for quality and safety:

- **Quality Metrics**: Grammar Correctness, Response Tone, Answer Relevance, Conceptual Similarity, Factual Consistency, Clarity, Coherence, Creativity, Readability, Diversity, Narrative Continuity
- **Safety & Content Moderation**: Toxicity, NSFW, Hate Speech, Violence, Terrorism, Self Harm, Sexual Content
- **Bias Detection**: Bias, Gender Bias, Racial Bias, Political Bias
- **Security**: Data Leakage, Insecure Output
- **Scoring Metrics**: BLEU Score, ROUGE Score, METEOR Score, Cosine Similarity, Fuzzy Score, Compression Score

### Agentic Behavior Validators (8 total)

Evaluate AI agent performance:

- **Planning**: Plan Optimality, Plan Coherence
- **Tool Usage**: Tool Call Accuracy, Tool Failure Rate
- **Goal Achievement**: Agent Goal Accuracy, Intent Resolution
- **Conversation**: Topic Adherence, Fallback Rate

### RAG Grounding Validators (7 total)

Evaluate Retrieval-Augmented Generation systems:

- **Context Quality**: Context Relevance, Context Recall, Context Precision, Context Entities Recall
- **Response Quality**: Response Relevancy, Faithfulness, Noise Sensitivity

### MCP Security Validators (3 total)

Security validators for Model Context Protocol:

- Prompt Injection Detection
- Data Leakage Detection
- Insecure Output Detection

### Themes Classifier (1 total)

Classify text into themes and sub-themes.

### Composite Score API

Combined weighted evaluation across multiple validators with detailed breakdowns.

## Field Requirements

Different validators require different combinations of fields:

| Field Combination | Input Validators | Output Validators |
|-------------------|------------------|-------------------|
| `prompt` only | All input validators | - |
| `response` only | - | Most safety/bias/quality validators |
| `prompt` + `response` | - | Answer Relevance, Conceptual Similarity |
| `context` + `response` | - | Factual Consistency, Creativity, All scoring metrics |

## Support

For more information, visit the [main README](../README.md).

