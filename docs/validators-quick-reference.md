# Validators Quick Reference

A quick reference guide for all Disseqt SDK validators.

> **Note:** For advanced validators (Agentic, RAG, MCP, Themes, Composite), see [advanced-validators.md](./advanced-validators.md)

## Input Validators (14 total)

### Safety & Content Moderation

| Validator | Slug | Description |
|-----------|------|-------------|
| `ToxicityValidator` | toxicity | Detects toxic language |
| `NSFWValidator` | nsfw | Identifies NSFW content |
| `HateSpeechValidator` | hate-speech | Detects hate speech |
| `ViolenceValidator` | violence | Detects violent content |
| `TerrorismValidator` | terrorism | Detects terrorism-related content |
| `SelfHarmValidator` | self-harm | Detects self-harm content |
| `SexualContentValidator` | sexual-content | Detects sexual content |

### Bias Detection

| Validator | Slug | Description |
|-----------|------|-------------|
| `BiasValidator` | bias | General bias detection |
| `GenderBiasValidator` | gender-bias | Gender-based bias detection |
| `RacialBiasValidator` | racial-bias | Racial/ethnic bias detection |
| `PoliticalBiasValidator` | political-bias | Political bias detection |
| `IntersectionalityValidator` | intersectionality | Intersectional bias detection |

### Security

| Validator | Slug | Description |
|-----------|------|-------------|
| `InputPromptInjectionValidator` | prompt-injection | Detects prompt injection attacks |
| `InvisibleTextValidator` | invisible-text | Detects hidden/invisible text |

---

## Output Validators (30 total)

### Quality Metrics

| Validator | Slug | Required Fields | Description |
|-----------|------|-----------------|-------------|
| `GrammarCorrectnessValidator` | grammatical-correctness | response | Grammar correctness |
| `ResponseToneValidator` | response-tone | response | Tone appropriateness |
| `AnswerRelevanceValidator` | answer-relevance | prompt + response | Response relevance to query |
| `ConceptualSimilarityValidator` | conceptual-similarity | prompt + response | Semantic similarity |
| `FactualConsistencyValidator` | factual-consistency | context + response | Factual accuracy |
| `ClarityValidator` | clarity | response | Clarity of response |
| `CoherenceValidator` | coherence | response | Logical flow |
| `CreativityValidator` | creativity | context + response | Creativity level |
| `ReadabilityValidator` | readability | response | Ease of reading |
| `DiversityValidator` | diversity | response | Content diversity |
| `NarrativeContinuityValidator` | narrative-continuity | response | Narrative flow |

### Safety & Content Moderation

| Validator | Slug | Required Fields | Description |
|-----------|------|-----------------|-------------|
| `OutputToxicityValidator` | toxicity | response | Detects toxic language |
| `OutputNSFWValidator` | nsfw | response | Identifies NSFW content |
| `OutputHateSpeechValidator` | hate-speech | response | Detects hate speech |
| `OutputViolenceValidator` | violence | response | Detects violent content |
| `OutputTerrorismValidator` | terrorism | response | Detects terrorism content |
| `OutputSelfHarmValidator` | self-harm | response | Detects self-harm content |
| `OutputSexualContentValidator` | sexual-content | response | Detects sexual content |

### Bias Detection

| Validator | Slug | Required Fields | Description |
|-----------|------|-----------------|-------------|
| `OutputBiasValidator` | bias | response | General bias detection |
| `OutputGenderBiasValidator` | gender-bias | response | Gender-based bias |
| `OutputRacialBiasValidator` | racial-bias | response | Racial/ethnic bias |
| `OutputPoliticalBiasValidator` | political-bias | response | Political bias |

### Security

| Validator | Slug | Required Fields | Description |
|-----------|------|-----------------|-------------|
| `OutputDataLeakageValidator` | data-leakage | response | Detects data leakage |
| `OutputInsecureOutputValidator` | insecure-output | response | Detects insecure code |

### Scoring Metrics

| Validator | Slug | Required Fields | Description |
|-----------|------|-----------------|-------------|
| `BleuScoreValidator` | bleu-score | context + response | BLEU scoring |
| `RougeScoreValidator` | rouge-score | context + response | ROUGE scoring |
| `MeteorScoreValidator` | meteor-score | context + response | METEOR scoring |
| `CosineSimilarityValidator` | cosine-similarity | context + response | Cosine similarity |
| `FuzzyScoreValidator` | fuzzy-score | context + response | Fuzzy matching |
| `CompressionScoreValidator` | compression-score | context + response | Compression ratio |

---

## Field Mapping

| SDK Field | API Field | Description |
|-----------|-----------|-------------|
| `prompt` | `llm_input_query` | User's input query/prompt |
| `context` | `llm_input_context` | Context/reference information |
| `response` | `llm_output` | LLM's output/response |

---

## Common Config Labels

### Risk-Based Labels
```python
custom_labels=["Safe", "Moderate Risk", "High Risk"]
label_thresholds=[0.3, 0.6]
```

### Bias Labels
```python
custom_labels=["Unbiased", "Biased", "Highly Biased"]
label_thresholds=[0.4, 0.7]
```

### Quality Labels
```python
custom_labels=["Poor", "Good", "Excellent"]
label_thresholds=[0.5, 0.8]
```

### Toxicity Labels
```python
custom_labels=["Non Toxic", "Toxic", "Highly Toxic"]
label_thresholds=[0.5, 0.75]
```

### Binary Labels
```python
custom_labels=["Safe", "Unsafe"]
label_thresholds=[0.5]
```

---

## Quick Code Examples

### Input Validation
```python
from disseqt_sdk.models import InputValidationRequest, SDKConfigInput
from disseqt_sdk.validators.input import ToxicityValidator

data = InputValidationRequest(prompt="Your text here")
config = SDKConfigInput(threshold=0.5)
validator = ToxicityValidator(data=data, config=config)
```

### Output Validation (Response Only)
```python
from disseqt_sdk.models import OutputValidationRequest, SDKConfigInput
from disseqt_sdk.validators.output import GrammarCorrectnessValidator

data = OutputValidationRequest(response="Your LLM response here")
config = SDKConfigInput(threshold=0.6)
validator = GrammarCorrectnessValidator(data=data, config=config)
```

### Output Validation (Prompt + Response)
```python
from disseqt_sdk.models import OutputValidationRequest, SDKConfigInput
from disseqt_sdk.validators.output import AnswerRelevanceValidator

data = OutputValidationRequest(
    prompt="What is the capital of France?",
    response="The capital of France is Paris."
)
config = SDKConfigInput(threshold=0.4)
validator = AnswerRelevanceValidator(data=data, config=config)
```

### Output Validation (Context + Response)
```python
from disseqt_sdk.models import OutputValidationRequest, SDKConfigInput
from disseqt_sdk.validators.output import FactualConsistencyValidator

data = OutputValidationRequest(
    context="Water's chemical formula is H2O.",
    response="Water is made of hydrogen and oxygen."
)
config = SDKConfigInput(threshold=0.6)
validator = FactualConsistencyValidator(data=data, config=config)
```

---

## Import Paths

```python
# Input validators
from disseqt_sdk.validators.input import (
    ToxicityValidator,
    BiasValidator,
    GenderBiasValidator,
    RacialBiasValidator,
    PoliticalBiasValidator,
    IntersectionalityValidator,
    NSFWValidator,
    HateSpeechValidator,
    ViolenceValidator,
    TerrorismValidator,
    SelfHarmValidator,
    SexualContentValidator,
    InputPromptInjectionValidator,
    InvisibleTextValidator,
)

# Output validators
from disseqt_sdk.validators.output import (
    # Quality
    GrammarCorrectnessValidator,
    ResponseToneValidator,
    AnswerRelevanceValidator,
    ConceptualSimilarityValidator,
    FactualConsistencyValidator,
    ClarityValidator,
    CoherenceValidator,
    CreativityValidator,
    ReadabilityValidator,
    DiversityValidator,
    NarrativeContinuityValidator,
    # Safety
    OutputToxicityValidator,
    OutputNSFWValidator,
    OutputHateSpeechValidator,
    OutputViolenceValidator,
    OutputTerrorismValidator,
    OutputSelfHarmValidator,
    OutputSexualContentValidator,
    # Bias
    OutputBiasValidator,
    OutputGenderBiasValidator,
    OutputRacialBiasValidator,
    OutputPoliticalBiasValidator,
    # Security
    OutputDataLeakageValidator,
    OutputInsecureOutputValidator,
    # Scoring
    BleuScoreValidator,
    RougeScoreValidator,
    MeteorScoreValidator,
    CosineSimilarityValidator,
    FuzzyScoreValidator,
    CompressionScoreValidator,
)
```

---

## Advanced Validators Quick Reference

### Agentic Behavior Validators (8 total)

| Validator | Slug | Description |
|-----------|------|-------------|
| `TopicAdherenceValidator` | topic-adherence | Topic consistency |
| `ToolCallAccuracyValidator` | tool-call-accuracy | Tool call accuracy |
| `ToolFailureRateValidator` | tool-failure-rate | Tool failure rate |
| `PlanOptimalityValidator` | plan-optimality | Plan efficiency |
| `AgentGoalAccuracyValidator` | agent-goal-accuracy | Goal achievement |
| `IntentResolutionValidator` | intent-resolution | Intent handling |
| `PlanCoherenceValidator` | plan-coherence | Plan logic |
| `FallbackRateValidator` | fallback-rate | Fallback frequency |

### RAG Grounding Validators (7 total)

| Validator | Slug | Description |
|-----------|------|-------------|
| `ContextRelevanceValidator` | context-relevance | Context-query relevance |
| `ContextRecallValidator` | context-recall | Context in response |
| `ContextPrecisionValidator` | context-precision | Context precision |
| `ContextEntitiesRecallValidator` | context-entities-recall | Entity preservation |
| `NoiseSensitivityValidator` | noise-sensitivity | Noise handling |
| `ResponseRelevancyValidator` | response-relevancy | Response relevance |
| `FaithfulnessValidator` | faithfulness | Faithfulness check |

### MCP Security Validators (3 total)

| Validator | Slug | Description |
|-----------|------|-------------|
| `McpPromptInjectionValidator` | prompt-injection | Injection detection |
| `DataLeakageValidator` | data-leakage | Leakage detection |
| `InsecureOutputValidator` | insecure-output | Insecure content |

### Themes & Composite

| Validator | Slug | Description |
|-----------|------|-------------|
| `ClassifyValidator` | classify | Theme classification |
| `CompositeScoreEvaluator` | evaluate | Combined evaluation |

### Advanced Imports

```python
# Agentic Behavior
from disseqt_sdk.validators.agentic_behavior import (
    TopicAdherenceValidator, ToolCallAccuracyValidator,
    ToolFailureRateValidator, PlanOptimalityValidator,
    AgentGoalAccuracyValidator, IntentResolutionValidator,
    PlanCoherenceValidator, FallbackRateValidator,
)

# RAG Grounding
from disseqt_sdk.validators.rag_grounding import (
    ContextRelevanceValidator, ContextRecallValidator,
    ContextPrecisionValidator, ContextEntitiesRecallValidator,
    NoiseSensitivityValidator, ResponseRelevancyValidator,
    FaithfulnessValidator,
)

# MCP Security
from disseqt_sdk.validators.mcp_security import (
    McpPromptInjectionValidator, DataLeakageValidator,
    InsecureOutputValidator,
)

# Themes & Composite
from disseqt_sdk.validators.themes_classifier import ClassifyValidator
from disseqt_sdk.validators.composite import CompositeScoreEvaluator

# Request Models
from disseqt_sdk.models import (
    AgenticBehaviourRequest, RagGroundingRequest,
    McpSecurityRequest, ThemesClassifierRequest,
    CompositeScoreRequest,
)
```

