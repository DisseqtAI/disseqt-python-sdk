# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Bumped `requests` to `>=2.33.0`, `pytest` to `>=9.0.3`, and `black` to
  `>=26.3.1` (also updating the pinned pre-commit `black` hook), and
  regenerated `uv.lock` to pull in patched `urllib3`, `filelock`, `idna`,
  `Pygments`, and `virtualenv`. Closes 12 open Dependabot alerts
  (DoS/ReDoS, TOCTOU symlink races, and cross-origin header forwarding).

## [0.7.0] - 2026-07-03

### Added
- **`Client.validate(request, policies=[...])`** — realtime policies are
  now evaluated straight from `validate()`. Three call shapes:
  1. `validate(SomeValidator(...))` — unchanged classic behavior.
  2. `validate(SomeValidator(...), policies=["<id>", ...])` — the
     validator runs as usual **and** the same input is evaluated against
     each policy server-side (each policy's own rulesets, thresholds, and
     decision strategy; one Decisions-ledger entry per policy).
  3. `validate(InputValidationRequest(...), policies=["<id>"])` — a bare
     request object (any `disseqt_sdk.models` request) with no validator
     and no config; the policies decide everything.
  With `policies` the return value is a stable envelope
  `{"validation": {...}|None, "policies": [<policy envelope>, ...]}`
  (policies evaluated sequentially, in order). Requires
  `Client(application_name=...)`.
- **`any_blocking(result)`** helper — True when any policy decision in
  the envelope (or a list of envelopes, or a single envelope) is BLOCK.
  Safe on classic validator responses (returns False).
- **`Client(policies=[...])`** — a client-level default policy list.
  Every `validate()` call evaluates it unless the call passes its own
  `policies=` (per-call always overrides). `[]` at construction means
  "no default"; an explicit per-call `policies=[]` still raises so an
  accidentally empty list fails loudly instead of silently ungating.
  Composite-score / themes-classifier requests run classically — the
  default steps aside for them (logged). Requires `application_name`.
- **`PolicyDecision.aggregation` / `.aggregate_score` / `.aggregate_threshold`**
  — `parse_policy` now surfaces the decision strategy that produced the
  verdict (`any | all | majority | weighted`) and, for weighted policies,
  the policy confidence and the blocking line it was compared against
  (`score >= threshold → BLOCK`). Empty/None against servers that predate
  aggregation enforcement. Documented in
  [docs/realtime-policies.md → Decision strategies](docs/realtime-policies.md#decision-strategies),
  including the skip-renormalization contract for partially-matched inputs.
- **Per-span / per-trace policy override (agentic SDK)** — a
  `realtime_policy_id` argument on `DisseqtTrace.start_span(...)`,
  `start_trace(...)`, and the `trace_llm_call` / `trace_tool_call` /
  `trace_agent_action` helpers, plus `realtime_policy_id` (span) and
  `trace_realtime_policy_id` (trace) on the `@trace_function` decorator.
  Precedence is **span override → trace override → client default → no
  policy**; `realtime_policy_id=""` opts a specific span out of evaluation.
  The transport buckets spans by effective policy id — one POST per
  distinct policy in a batch.
- **`AgenticBehaviourRequest` now carries the LLM text fields too** —
  `prompt` / `context` / `response` (serialized to `llm_input_query` /
  `llm_input_context` / `llm_output`) alongside the agentic arrays
  (`conversation_history`, `tool_calls`, `agent_responses`,
  `reference_data`). This makes it the single carrier for a policy that
  mixes text-domain and agentic validators in one `validate(...,
  policies=[...])` call.

### Removed
- **`Client.evaluate_policy()`** and the **`Client(realtime_policy_id=...)`**
  constructor parameter (introduced in 0.6.0). Policy evaluation is now
  exclusively `validate(..., policies=[...])` — same endpoint, same
  envelope per policy, plus the ability to run a validator alongside.
  Migration:

  ```python
  # before (0.6.0)
  client.evaluate_policy(realtime_policy_id=PID, prompt=text)
  # after
  client.validate(InputValidationRequest(prompt=text), policies=[PID])["policies"][0]
  ```

  Note: unknown/unpublished policies answer HTTP 404 (DSQ-4040) since
  production-monitoring v0.1.12 — branch on `e.status_code == 404`.

## [0.6.0] - 2026-07-02

### Added
- **Realtime policy evaluation** (`disseqt_sdk`): new
  `Client.evaluate_policy(realtime_policy_id, prompt=, context=, response=,
  conversation_history=, tool_calls=, agent_responses=, reference_data=,
  ...)` runs a published policy server-side and returns a structured
  verdict with per-rule breakdown. The typed kwargs are renamed to the
  wire shape the validators expect (`prompt` → `llm_input_query`, etc.);
  pass `input_data=` as a raw dict only for shapes the typed args don't
  cover. Caller helpers `is_blocking(result)` and `is_async(result)` plus
  `parse_policy(result)` returning typed `PolicyDecision` / `PolicyRuleset`
  / `PolicyRule` dataclasses.
- **`Client.realtime_policy_id`** + **`Client.application_name`**: optional
  defaults for `evaluate_policy()`. `application_name` is required
  whenever `realtime_policy_id` is set (matches `service_name` on the
  agentic SDK). Per-call value always wins over the Client default.
- **`Client.realtime_policy_base_url`**: separate base URL for the policy
  evaluate endpoint (defaults to
  `https://api.disseqt.ai/realtime-validations` — the evaluate endpoint
  is served by production-monitoring next to the validators; the
  `/realtime-policies` gateway is the policy CRUD dashboard and has no
  SDK routes). Independent of `base_url` so the two endpoints can be
  mocked / routed independently — override for local testing (e.g.
  `http://localhost:9010`).
- **Agentic SDK realtime policies**: `DisseqtAgenticClient(realtime_policy_id=)`
  sets a default policy stamped on every span's resource block as
  `policy.id`. `start_trace(client, name, realtime_policy_id=)` overrides
  per-trace — lets two agents inside one application run under different
  policies without re-initialising the client. The transport groups
  buffered spans by policy and emits one POST per distinct policy.

### Changed
- **`enforcement` field** in policy responses now mirrors the policy's
  `strategy.executionMode` (values: `"sync"` / `"async"`) — decoupled
  from the BLOCK/PASS verdict, which is in `decision`. Previously this
  field held `"blocking"` / `"advisory"`.

### Fixed
- `examples/example_composite_score.py`: removed trailing commas that
  turned `PROJECT_ID` and `API_KEY` into single-element tuples.
- `INSTALL.md`, `docs/README.md`, `docs/validators.md`,
  `examples/verify_installation.py`: fixed `pip install disseqt-sdk` →
  `pip install disseqt-ai-sdk` (actual PyPI name), and
  `DisseqtClient(...)` → `Client(project_id=..., api_key=...)` in
  example code (the class name and required args were wrong).

## [0.5.0] - 2026-07-01

### Added
- **Built-in structured logging** via a new dependency-free `disseqt_logging`
  package (standard library only). Provides JSON/console output, automatic
  `service`/`env`/`host` fields, PII/credential redaction (email/JWT/card/phone/
  token shapes + a sensitive-key deny-list), a privacy-safe `digest`, an error
  envelope, and a dynamic level. **Silent by default** — emits nothing until you
  opt in via `disseqt_sdk.configure_logging(...)` / `set_log_level(...)` or
  `DISSEQT_LOG_LEVEL`, so existing installs see no new output. `disseqt_logging.disable()`
  silences it again.
- **`Client.validate()` instrumentation**: structured `validation.request` /
  `validation.response` / `validation.http_error` / `validation.network_error`
  events with latency and a content-free payload digest. Request payloads, auth
  headers, `api_key`, and `project_id` are never logged.

### Changed
- **Agentic SDK logging** (`disseqt_agentic_sdk.utils.logging`) now routes
  through the shared `disseqt_logging` logger. `get_logger()` still returns a
  standard-library `logging.Logger` (unchanged type — `isinstance`, `setLevel`,
  `addHandler`, etc. keep working); only the output format becomes structured and
  it is silent until enabled. `set_log_level` is unchanged.
- `DisseqtAgenticClient` initialization no longer includes `project_id` in its
  log fields (defense in depth — the value is never emitted even with redaction
  disabled).

## [0.4.0] - 2026-06-03

### Added
- **Intent validators** (both directions): `intent-guard` (per-project BLOCK list;
  response `enforcement` "blocking") and `intent-compliance` (per-project ALLOW
  list; `enforcement` "advisory"), exposed as `IntentGuardValidator` /
  `IntentComplianceValidator` (input — evaluates the prompt) and
  `OutputIntentGuardValidator` / `OutputIntentComplianceValidator` (output —
  evaluates the response). Same `client.validate(...)` flow as the other validators.
- **`SDKConfigInput.intents`**: optional `list[str]` carried inside `config_input`
  for the intent validators. An empty/omitted list defers to the project's
  dashboard-configured intent list.

## [0.3.0] - 2026-05-26

### Changed
- **Default validation API endpoint**: `Client` default `base_url` updated from
  `https://production-monitoring-eu.disseqt.ai` to
  `https://api.disseqt.ai/realtime-validations`. Callers using the default will
  now target the new endpoint on upgrade. Pass `base_url=` explicitly to opt out.
  Examples, tests, and docs updated to match.

## [0.2.0] - 2025-12-01

### Changed
- **Python Version Requirement**: Updated minimum Python version to >=3.10.14 (from >=3.12)
  - Added Python 3.10 and 3.11 to supported classifiers
  - Updated tool configurations (Black, Ruff, Mypy) to target Python 3.10
  - Rebuilt UV environment with Python 3.10.14
  - All tests passing on Python 3.10.14
  - Broader compatibility for users
- **Project Organization**: Moved all example files to `examples/` directory
  - Created `examples/` folder with comprehensive documentation
  - Better project structure following Python best practices

### Added
- **🚀 COMPOSITE SCORE EVALUATOR**: New comprehensive evaluation system
  - Combines multiple validators into a single weighted score
  - Evaluates three main categories:
    * **Factual/Semantic Alignment**: 9 metrics (factual_consistency, answer_relevance, conceptual_similarity, compression_score, rouge_score, cosine_similarity, bleu_score, fuzzy_score, meteor_score)
    * **Language Quality**: 3 metrics (clarity, readability, response_tone)
    * **Safety/Security/Integrity**: 6 metrics (toxicity, gender_bias, racial_bias, hate_speech, data_leakage, insecure_output)
  - Features:
    * Custom weight configuration for top-level and submetric categories
    * Configurable label thresholds with custom labels
    * Binary threshold or weighted scoring modes
    * Detailed breakdown of passed/failed metrics per category
    * Overall confidence score with label
    * Credit tracking and usage information
  - New components:
    * `CompositeScoreRequest` model with `llm_input_query`, `llm_output`, `llm_input_context`
    * `CompositeScoreEvaluator` validator with custom request/response handlers
    * Dedicated endpoint: `/api/v1/validators/composite/evaluate`
    * `ValidatorDomain.COMPOSITE` enum
    * Example usage in `example_composite_score.py`

- **🎉 COMPLETE VALIDATOR IMPLEMENTATION**: Implemented all 52 core validators (81.25% of total)
  - **Input Validation**: 14/14 validators (100% COMPLETE) ✅
    - `ToxicityValidator`, `BiasValidator`, `InputPromptInjectionValidator` (existing)
    - `IntersectionalityValidator`, `RacialBiasValidator`, `GenderBiasValidator` (new)
    - `SelfHarmValidator`, `ViolenceValidator`, `TerrorismValidator` (new)
    - `SexualContentValidator`, `HateSpeechValidator`, `NSFWValidator`, `InvisibleTextValidator` (new)
  - **Agentic Behavior**: 9/9 validators (100% COMPLETE) ✅
    - `TopicAdherenceValidator`, `ToolCallAccuracyValidator` (existing)
    - `ToolFailureRateValidator`, `PlanOptimalityValidator`, `AgentGoalAccuracyValidator` (new)
    - `IntentResolutionValidator`, `PlanCoherenceValidator`, `FallbackRateValidator` (new)
  - **MCP Security**: 3/3 validators (100% COMPLETE) ✅
    - `McpPromptInjectionValidator`, `DataLeakageValidator` (existing)
    - `InsecureOutputValidator` (new)
  - **Themes Classifier**: 1/1 validators (100% COMPLETE) ✅
    - `ClassifyValidator` with custom request/response handlers
  - **RAG Grounding**: 7/8 validators (87.5% complete)
    - `ContextRelevanceValidator`, `FaithfulnessValidator` (existing)
    - `ContextRecallValidator`, `ContextPrecisionValidator`, `ResponseRelevancyValidator` (new)
    - `ContextEntitiesRecallValidator`, `NoiseSensitivityValidator` (new)
  - **Output Validation**: 14/25 validators (56% complete)
    - `FactualConsistencyValidator`, `AnswerRelevanceValidator`, `ClarityValidator`, `OutputToxicityValidator` (existing)
    - `OutputBiasValidator`, `CoherenceValidator`, `OutputDataLeakageValidator`, `OutputInsecureOutputValidator` (new)
    - `BleuScoreValidator`, `RougeScoreValidator`, `MeteorScoreValidator` (new)
    - `CosineSimilarityValidator`, `FuzzyScoreValidator`, `CompressionScoreValidator` (new)

- **Registry Pattern Enhancement**: Enhanced `@register_validator` decorator with optional custom handlers
  - `request_handler`: Custom request payload formatting per validator
  - `response_handler`: Custom response processing per validator
  - Backward compatible with existing validators
- **Flexible Response Handling**: No forced normalization, preserves API response structure
- **Enhanced Enums**: Added 40+ new validator slugs across all domains
- **Comprehensive Test Suite**: 97% test coverage achieved (exceeds >95% target)
  - 127 total tests (51 new tests added)
  - Full coverage of composite score feature
  - Client integration tests with header verification
  - Edge case testing (unicode, special characters, empty values)
  - Error handling tests (HTTP errors, network errors)
  - All validator post-init methods tested
- **Examples Organization**: Created `examples/` directory with documentation
  - `example.py` - General validator usage examples
  - `example_composite_score.py` - Composite score evaluation examples
  - `verify_installation.py` - Installation verification utility
  - `examples/README.md` - Comprehensive examples documentation

### Changed
- **Path Template Standardization**: Unified to `/api/v1/sdk/validators/{domain}/{validator}`
- **Response Architecture**: Moved from centralized normalization to validator-specific handlers
- **Registry System**: Enhanced to support custom request/response processing per validator
- **Import Structure**: Organized validators by domain with proper `__init__.py` imports

### Fixed
- **URL Path Construction**: Removed extra `/validators` segment from API endpoints
- **Test Compatibility**: All 76 tests passing with new validator implementations
- **Enum Completeness**: All validator slugs properly defined in domain enums
- **Import Errors**: Resolved circular imports and missing enum attributes

### Implementation Status
- **Total Progress**: 52/64 validators (81.25% complete) 🚀
- **Completed Domains**: 4/6 domains at 100%
  - ✅ Input Validation (14/14)
  - ✅ Agentic Behavior (9/9)
  - ✅ MCP Security (3/3)
  - ✅ Themes Classifier (1/1)
- **Nearly Complete**: RAG Grounding (7/8, missing only `answer-correctness`)
- **Major Progress**: Output Validation (14/25, core metrics implemented)

### 🎯 **MAJOR MILESTONE ACHIEVED**
- **All Core Safety Validators**: Complete coverage of toxicity, bias, hate speech, violence, terrorism, self-harm detection
- **All Agentic Behavior Validators**: Complete coverage of tool accuracy, plan optimality, goal accuracy, intent resolution
- **All Security Validators**: Complete coverage of prompt injection, data leakage, insecure output detection
- **Production Ready**: SDK now supports 52 validators with robust, extensible architecture

### Architecture Highlights
- **Registry Pattern**: Flexible decorator-based registration with custom handlers
- **Type Safety**: Full type hints with Python 3.12.5 compatibility
- **Request/Response Flexibility**: Each validator can define custom API interaction patterns
- **Backward Compatibility**: Existing code continues to work unchanged
- **Extensible Design**: Easy addition of remaining 12 validators (specialized NLP metrics)

## [0.1.0] - 2025-10-30

### Added
- Initial SDK implementation with core architecture
- Base validator classes and domain-specific subclasses
- Client with authentication and error handling
- Registry system for dynamic validator discovery
- Comprehensive test suite with 76 tests
- Documentation and development tooling
