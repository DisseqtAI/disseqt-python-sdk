# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.1] - 2026-08-24

### Changed
- **BREAKING** — ``application_id`` is now a **required, keyword-only**
  argument to ``DisseqtAgenticClient(...)``. Missing raises
  ``TypeError`` at Python's argument-binding stage; empty /
  whitespace-only raises ``ValueError`` at construction. Kong's
  traces-auth plugin drops every trace POST without a matching
  ``X-Application-Id`` header, so constructing a client that will
  never deliver spans is now impossible. Migration: add
  ``application_id="your-application-uuid"`` to every
  ``DisseqtAgenticClient(...)`` call — obtain the value from the
  Applications Registry (URL in the ``ValueError`` message).

### Removed
- ``disseqt_agentic_sdk._notices`` module and the
  ``DISSEQT_SDK_DISABLE_APPLICATION_ID_NOTICE`` env var — the
  missing-application_id WARNING notice (introduced in 0.10.0 as an
  interim nudge) is dead code once the field is required.

## [0.11.0] - 2026-08-24

_Everything below was originally staged under a hand-bumped 0.10.0 that
was never tagged or published — there is no 0.10.0 on PyPI. Its changes
ship in this release._

### Added
- **Auto-instrumentation for popular LLM SDKs** — one
  `instrument_all(client)` call at startup patches every installed
  provider SDK we support so every LLM call emits a `MODEL_EXEC` span
  automatically, with no wrapping in user code. Supported providers
  (minimum versions in parens): OpenAI (>=1.50), Anthropic (>=0.40),
  Groq (>=0.11), Mistral / `mistralai` (>=1.5), Cohere v2 (>=5.11),
  Google Gemini via `google-genai` (>=1.0), and LiteLLM (>=1.40). Sync,
  async, and streaming covered for each. Spans dual-emit `agentic.*`
  and OpenTelemetry `gen_ai.*` attributes so traces are consumable by
  OTel-native tooling without translation. Selective use via
  `instrument("openai", client)`; disable via `uninstrument("openai")`
  / `uninstrument_all()`. Install per-provider or all-at-once extras:
  `pip install "disseqt-ai-sdk[openai,anthropic]"` or
  `pip install "disseqt-ai-sdk[instrumentation]"`.
- **Auto-capture of LLM tool calls (Lane A)** — every MODEL_EXEC span
  now carries `agentic.tool_calls` + `gen_ai.tool_calls` normalized to
  a canonical `[{id, name, arguments (JSON str)}]` shape across all
  four provider tool-call formats (OpenAI, Anthropic, Gemini,
  Cohere v2). Streaming coverage included. Request-side `tools=[...]`
  schema captured as `agentic.request.tools`.
- **`agent_span` + `record_tool_result` (Lane B)** — context manager
  that opens an AGENT_EXEC span and aggregates planned tool calls
  (from nested MODEL_EXEC spans) with the user-supplied execution
  outcomes (`success` / `failure` / `error` / `timeout`) into a fused
  `agentic.tool_calls` list on the AGENT_EXEC span — which is what
  the tool-failure-rate, tool-call-accuracy, plan-optimality, and
  plan-coherence validators read. Async-safe via contextvars.
- **OpenAI Batch API instrumentation** — `client.batches.create()` /
  `.retrieve()` / `.cancel()` (sync + async) each emit a MODEL_EXEC
  span tagged with a shared `agentic.batch.id` so downstream can
  reconstruct the create → poll → complete lifecycle by GROUP BY.
  Adds a canonical batch shape + adapter layer ready for Anthropic /
  Mistral batch follow-ups.
- **Embeddings-specific attributes** — `agentic.embeddings.*` on every
  OpenAI embeddings call: `input_count`, `dimensions_requested`,
  `dimensions_actual` (measured from response), `encoding_format`,
  `count`, plus `agentic.request.user`. Canonical adapter layer in
  place for adding Mistral / Cohere / Gemini / LiteLLM embeddings.
- **User-supplied custom attributes on auto-spans** — new
  `set_span_attributes(**kwargs)`, `clear_span_attributes()`, and
  `span_context(**kwargs)` helpers. Attributes set in the current
  context are merged onto every auto-created span at scope-exit time,
  **overriding auto values on key collision** (user intent always
  wins). Backed by `contextvars.ContextVar` — async-safe, no
  bleed-through between concurrent tasks.
- **Lifecycle hooks on instrumentors** — optional `on_install` /
  `on_uninstall` callbacks on `instrument()` / `instrument_all()` /
  `uninstrument()` / `uninstrument_all()`. Callbacks fire outside the
  registry lock; exceptions from user hooks are swallowed so bad
  observability plumbing can't corrupt instrumentation state.
- **Structured instrumentation errors** — new `InstrumentationError`
  with a stable `.reason` string (`unknown_provider`, `load_failure`,
  `package_missing`, `unsupported_version`, `already_instrumented`,
  `client_mismatch`, `instrument_failure`). Opt in via `strict=True`
  on `instrument()` / `instrument_all()`; non-strict path keeps the
  existing bool return but logs the specific reason.
- **Duration tracking + slow-call warnings** — every auto span emits
  `agentic.request.duration_ms`, and a WARNING logs when the wrapped
  call exceeds a configurable threshold (default 5 min). Configure
  via `set_slow_call_threshold_ms(ms)`; pass `None` to disable.
- **`get_instrumented_client(name)`** — new public helper returns the
  client currently bound to a provider, or `None` if not instrumented.
- **`application_id` recommendation notice** — `DisseqtAgenticClient(...)`
  now logs a one-shot `WARNING` through the stdlib `disseqt_agentic_sdk`
  logger when constructed without `application_id`, pointing to the AI
  Applications Registry docs. Purely informational; ingest behaviour is
  unchanged. Suppress with `DISSEQT_SDK_DISABLE_APPLICATION_ID_NOTICE=1`
  or `logging.getLogger("disseqt_agentic_sdk").setLevel(logging.ERROR)`.

### Fixed
- **Thread-safe `_ACTIVE` registry** — concurrent `instrument_all()`
  calls from multiple threads could race the check-then-set and
  double-patch. Now guarded by an `RLock`.
- **Uninstrument no longer leaks client references** — long-running
  processes that repeatedly instrument/uninstrument accumulated a
  live client per cycle via wrapper closures. `_client` is now
  cleared on scope exit.
- **Robust unwrap + rollback on partial-instrument failure** — if
  `_instrument()` raised after patching some methods, those patches
  were left installed with no tracking entry. Now unwound. Also
  `_restore_wrapped` uses identity checks so we no longer tear down
  another library's wrapper when they've stacked on top of ours.
- **Client-mismatch detection** — calling `instrument("openai", B)`
  after `instrument("openai", A)` used to silently return `False`
  with a debug log; now warns loudly and refuses the rebind unless
  `uninstrument()` is called first.
- **Defensive attribute-writer wrapping** — bugs in our attribute
  writers (e.g. from malformed provider responses) can no longer
  crash the user's LLM call. All `set_common_chat_request` /
  `set_chat_response` / `_set_request_attrs` / `_set_response_attrs`
  / `set_batch_attrs` / `_set_embeddings_*` invocations go through
  `safe_call` — log-and-continue on error.

### Refactor
- Extracted request-kwarg string keys (`"stream"`, `"model"`,
  `"messages"`, `"prompt"`, `"input"`, `"system"`, `"contents"`,
  `"config"`, `"tools"`) into named constants in `_kwargs.py`.
- Type-annotated every wrapt wrapper across the 7 provider modules
  and the context-manager exits; removed all
  `# type: ignore[no-untyped-def]` from the instrumentation tree.
- Standardized best-effort error handling: `contextlib.suppress` for
  pure swallow, `except Exception as e:` only where the exception
  message is needed to build a structured error.
- Filled in skimpy docstrings on `_oai_compat.read`, the
  `ChatStreamAccumulator` methods, `base._detect_version`, and
  `base._version_lt`.

### Dependencies
- Added `wrapt>=1.16.0` as a runtime dependency (used by the
  auto-instrumentation monkey-patcher).
- Added optional-dependencies extras for each supported provider:
  `openai`, `anthropic`, `groq`, `mistral`, `cohere`, `gemini`,
  `litellm`, and `instrumentation` (installs all seven).

## [0.9.0] - 2026-08-22

### Added
- **`X-SDK-Lang: python` identity header** — names this SDK's release
  line so the backend's per-language version config compares it against
  the Python floor, never the Node/Go one. (The middleware assumes
  Python when the header is absent, so 0.8.0 clients stay correct.)
- **`SDKVersionBlockedError`** — when the server refuses a call with
  HTTP 426 (DSQ-4260, the SDK version enforcement tier — a permanent
  cutoff or a scheduled brownout rehearsal), the SDK now raises this
  typed exception instead of a generic `HTTPError`. It subclasses
  `HTTPError`, so existing `except HTTPError` handlers keep working
  unchanged; catch the new type to branch specifically on "upgrade
  required". Carries `.latest`, `.notice`, and `.sunset` (RFC 8594
  cutoff date) parsed from the response, and its message is the
  server's self-explanatory refusal text. Raised from all request
  paths (validator, policy-evaluate, Prompt Packs). `HTTPError` and
  `SDKVersionBlockedError` are now exported from the package root.

## [0.8.0] - 2026-08-12

### Added
- **SDK version notification** — every API call now identifies the SDK
  build via `X-SDK-Version` and `User-Agent: disseqt-ai-sdk/<version>`
  request headers (validator, policy-evaluate, and Prompt Packs
  clients). When the server advertises a newer release on the response
  (`X-SDK-Latest-Version`, plus `X-SDK-Notice` below the supported
  floor), the SDK logs one `WARNING` through the stdlib `disseqt_sdk`
  logger — at most once per process per advertised version, fail-open
  (a malformed or missing header can never affect a call), with zero
  extra network requests. Opt out of the warning with
  `DISSEQT_SDK_DISABLE_VERSION_NOTICE=1` (read once at import; the
  request headers are still sent). Silence it instead with
  `logging.getLogger("disseqt_sdk").setLevel(logging.ERROR)`.

### Fixed
- **Single-sourced `__version__`** — both `disseqt_sdk.__version__` and
  `disseqt_agentic_sdk.__version__` (and
  `DisseqtAgenticClient.SDK_VERSION`) now resolve the installed
  `disseqt-ai-sdk` distribution version via `importlib.metadata`
  (`0.0.0-dev` on an uninstalled source checkout), fixing the agentic
  package's stale hardcoded `0.1.0`.

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
