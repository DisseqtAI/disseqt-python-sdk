# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`disseqt-ai-sdk` is a Python SDK (v0.2.7) for Disseqt AI containing two independent sub-packages:

- **`disseqt_sdk`** — Validation SDK: validates LLM inputs, outputs, RAG grounding, agentic behavior, MCP security, and composite scoring via a remote API
- **`disseqt_agentic_sdk`** — Agentic SDK: OpenTelemetry-compatible tracing for agentic AI workflows (spans, traces, batching)

Both packages ship in a single PyPI distribution (`pip install disseqt-ai-sdk`), built with Hatch, and live under `src/`.

## Commands

```bash
# Setup
uv sync                              # install all deps (including dev group)
uv run pre-commit install            # install pre-commit hooks (black, ruff, trailing-whitespace, etc.)

# Testing
uv run pytest                        # run all tests (verbose, strict markers)
uv run pytest tests/unit/            # unit tests only
uv run pytest tests/agentic/         # agentic SDK tests only
uv run pytest tests/integration/     # integration tests (requires live API)
uv run pytest -k test_name           # run a single test by name
uv run pytest -q --cov=src --cov-report=term-missing  # with coverage

# Linting & Formatting
uv run ruff check .                  # lint
uv run ruff check . --fix            # lint with auto-fix
uv run black --check .               # format check
uv run black .                       # format
uv run mypy                          # type checking (strict mode configured in pyproject.toml)

# Build & Publish
uv build                             # build wheel + sdist
# Publishing happens via GitHub Actions on release (trusted publishing to PyPI)

# Run examples
python examples/run_all_examples.py  # runs all example scripts
```

## Architecture

### Validation SDK (`src/disseqt_sdk/`)

The validation SDK uses a **decorator-based registry pattern**:

1. **Enums** (`enums.py`) — `ValidatorDomain` maps to API route segments (e.g., `input-validation`). Each domain has its own enum for validator slugs (e.g., `InputValidation.TOXICITY`).
2. **Registry** (`registry.py`) — `@register_validator(domain, slug)` decorator stores validator metadata in a global `_VALIDATOR_REGISTRY` dict keyed by `"domain:slug"`.
3. **Base validators** (`validators/base.py`) — Domain-specific base classes (`InputValidator`, `OutputValidator`, `RagGroundingValidator`, `AgenticBehaviourValidator`, `McpSecurityValidator`, `ThemesClassifierValidator`) each carry a typed `data` request object and implement `to_payload()`.
4. **Request models** (`models/`) — Frozen dataclasses per domain: `InputValidationRequest`, `OutputValidationRequest`, `RagGroundingRequest`, `AgenticBehaviourRequest`, `McpSecurityRequest`, `CompositeScoreRequest`, `ThemesClassifierRequest`.
5. **Client** (`client.py`) — `Client.validate(request)` resolves the URL via `routes.py`, calls the remote API, and optionally applies custom request/response handlers from the registry.
6. **API Client** (`api_client.py`) — `DisseqtAPIClient` for Prompt Packs REST lifecycle (generate, runs, outputs, validations). Separate from the validator client.
7. **Response normalization** (`response.py`) — `normalize_server_payload()` standardizes API responses into a consistent `{data: {metric_name, actual_value, ...}, status: {...}}` envelope.
8. **Composite evaluator** (`validators/composite/evaluate.py`) — `CompositeScoreEvaluator` orchestrates multiple validators with weighted scoring.

### Agentic SDK (`src/disseqt_agentic_sdk/`)

OpenTelemetry-inspired tracing with:
- **Client** (`client/`) — `DisseqtAgenticClient` manages transport and batching
- **Trace/Span** (`trace/`, `span/`) — `DisseqtTrace` and `DisseqtSpan` with context manager support
- **Context** (`context/`) — Thread-local context for nested span hierarchies
- **Enums** (`enums/`) — `SpanKind` (MODEL_EXEC, TOOL_EXEC, AGENT_EXEC, RAG_EXEC, MCP_EXEC, etc.), `SpanStatus`
- **Semantics** (`semantics/`) — Agentic-specific attribute conventions
- **API helpers** (`api/`) — `start_trace()`, `trace_llm_call()`, `trace_tool_call()`, `trace_function()` decorators

### Adding a New Validator

1. Create a file in the appropriate `validators/<domain>/` directory
2. Subclass the domain's base validator (e.g., `InputValidator`)
3. Add `@register_validator(domain=..., slug=...)` decorator
4. Add the slug to the corresponding enum in `enums.py`
5. Import in the domain's `__init__.py`
6. Add tests in `tests/unit/`

## Key Conventions

- Python 3.10+ required; all function signatures must have type annotations
- Dataclasses with `slots=True` for validators and models
- `requests` library for HTTP (no async)
- Auth via `X-API-Key` and `X-Project-Id` headers
- API base URL defaults: validation SDK uses `api.disseqt.ai/realtime-validations`, Prompt Packs uses `localhost:8000`
- Line length: 100 (black + ruff)
- Pre-commit hooks: black, ruff (with `--fix`), trailing-whitespace, end-of-file-fixer, debug-statements
- `architecture/` directory is gitignored (local-only diagrams)
