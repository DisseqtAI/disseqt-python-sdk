# Realtime Policies

A **realtime policy** is a named, versioned bundle of validators — with their
thresholds, labels, and an enforcement strategy — that you author once in the
Disseqt dashboard and bind to your application by **policy id**. Once bound,
the policy governs every `client.validate(...)` call: it decides **which
validators run**, **at what thresholds**, and which are **skipped** — all
without a code deploy.

```
Client(realtime_policy_id="…")           # bind once
        │
        ▼
client.validate(SomeValidator(...))       # every call is policy-governed
        │
        ├─ validator enabled in policy → RUNS (policy threshold wins)
        └─ validator not in policy     → SKIPPED (no API call, no charge)
```

Change a threshold, enable a validator, disable another — publish a new policy
version in the dashboard, and every bound client picks it up within a minute.
The set of checks your application runs is owned by the dashboard, not by
your code.

---

## Table of contents

- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [How gating works](#how-gating-works)
- [The skip marker](#the-skip-marker)
- [Configuration precedence](#configuration-precedence)
- [Validator matching rules](#validator-matching-rules)
- [Caching and failure behavior](#caching-and-failure-behavior)
- [Discovering policies from code](#discovering-policies-from-code)
- [Agentic SDK: policy on every span](#agentic-sdk-policy-on-every-span)
- [`evaluate_policy()` — deprecated](#evaluate_policy--deprecated)
- [Error handling](#error-handling)
- [Production patterns](#production-patterns)
- [Configuration reference](#configuration-reference)

---

## Quick start

```python
from disseqt_sdk import Client, is_policy_skipped
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.validators.input.invisible_text import InvisibleTextValidator

client = Client(
    project_id="your_project_id",
    api_key="your_api_key",
    realtime_policy_id="994ad00e-bff4-4cff-8a45-2635f3f3fcd0",  # published policy
    application_name="checkout-bot",                            # required with a policy
)

# Policy passed == enabled. No threshold, no config — the policy owns
# the validator's configuration.
result = client.validate(
    InvisibleTextValidator(data=InputValidationRequest(prompt=user_input))
)

if is_policy_skipped(result):
    # This validator isn't enabled in the bound policy — nothing ran,
    # nothing was charged.
    ...
else:
    # Normal validation response, stamped with the governing policy:
    print(result["policy"])
    # {'policy_id': '994ad00e-…', 'policy_name': 'Prompt-Injection Defense',
    #  'policy_version': 1, 'enforcement': 'sync', 'threshold_source': 'policy'}
```

Your existing `validate()` code doesn't change — binding the policy on the
`Client` is the only difference. Without `realtime_policy_id`, `validate()`
behaves exactly as it always has.

---

## Prerequisites

1. **A published policy.** Create one in the dashboard under
   **Realtime Policies → Policies** (start from a template or build from
   scratch), then **Publish**. Copy the **Policy ID** (a UUID) shown on
   publish. Only *published* policies are visible to the SDK — drafts are
   invisible.
2. **A project API key** (`X-API-Key`) and **project id** (`X-Project-Id`) —
   the same credentials as every SDK call.
3. **`application_name`** — required whenever a policy id is set. It
   identifies your app on the dashboard. The constructor raises `ValueError`
   if you bind a policy without it.

---

## How gating works

When the client is bound to a policy, each `validate(SomeValidator(...))` call
goes through three steps:

1. **Resolve the policy** — the SDK fetches the policy definition from
   `GET /api/v1/sdk/policies/{id}` (cached for 60 seconds; see
   [Caching](#caching-and-failure-behavior)).
2. **Look up the validator** in the policy's rulesets (see
   [Matching rules](#validator-matching-rules)).
3. **Run or skip**:
   - **Enabled in the policy** → the validator runs as normal, except the
     policy's threshold replaces the code-level one. The response gains a
     `policy` block.
   - **Absent or disabled** → the call returns a skip marker immediately.
     **No HTTP request is made and no credits are spent.**

Composite score (`CompositeScoreEvaluator`) and themes-classifier requests are
**never gated** — they pass through unchanged regardless of the bound policy.

---

## The skip marker

A skipped call returns this shape instead of a validation response:

```python
{
    "skipped": True,
    "skipped_reason": "validator_not_in_policy",   # or "validator_disabled_in_policy"
    "validator_type": "input-validation",
    "validator_name": "toxicity",
    "policy": {
        "policy_id": "994ad00e-…",
        "policy_name": "Prompt-Injection Defense",
        "policy_version": 1,
        "enforcement": "sync",
    },
}
```

Branch on it with the helper rather than poking at keys:

```python
from disseqt_sdk import is_policy_skipped

if is_policy_skipped(result):
    log.info("validator %s not enabled in policy %s",
             result["validator_name"], result["policy"]["policy_id"])
```

Skips mirror the server's own skip semantics (the same way policy evaluation
marks rules `skipped` with a `skipped_reason`), and they're logged client-side
as `validation.policy_skip` events.

---

## Configuration precedence

**Every config key the policy provides wins.** A policy-governed run applies
the policy's per-validator configuration — `threshold`, `custom_labels`,
`label_scores` — over anything the code passes, with the exact per-key
precedence the server applies to `config_input` during full-policy
evaluation. Keys the policy doesn't set are left as the caller supplied
them. A caller cannot weaken (or tighten) dashboard-enforced guardrails
from code.

Because the policy owns the configuration, `config=` is **optional** on
every validator (and keyword-only): under a bound policy, pass only the
data. Standalone (no-policy) callers keep passing
`config=SDKConfigInput(threshold=…)` as before; when omitted entirely, the
threshold defaults to `0.5`.

The response tells you which threshold source applied:

```python
result["policy"]["threshold_source"]   # "policy" (normal) or "config" (policy had no threshold)
```

> `custom_labels` / `label_scores` are applied when the server exposes them
> in the policy detail (production-monitoring ≥ the PR #110 build); on older
> servers the gate applies the threshold only.

---

## Validator matching rules

How the SDK decides whether the validator you passed "is in" the policy:

1. **Canonical names** — the dashboard vocabulary uses underscores
   (`prompt_injection`), SDK slugs use hyphens (`prompt-injection`); they are
   compared in one canonical form, case-insensitively.
2. **Validator type must match the domain** — names alone are ambiguous
   (`prompt-injection` exists in both input-validation and mcp-security;
   most safety validators exist for input and output). A policy rule whose
   `validator_type` is set (the server always sets it) only matches a
   validator of the same domain — mirroring exactly which validator the
   *server* would run for that rule.
3. **`_output` suffix vocabulary** — the policy store disambiguates output
   variants by suffix: a rule named `hate_speech_output`
   (type `output-validation`) matches the SDK's output `hate-speech`
   validator.
4. **Enabled beats disabled** — if the same validator appears in several
   rulesets (disabled in one, enabled in another), the enabled entry wins,
   matching the server's evaluator which runs every ruleset.

If no rule matches, the call is skipped with `validator_not_in_policy`; if the
only matching rules are disabled, with `validator_disabled_in_policy`.

---

## Caching and failure behavior

The policy definition is cached **per client** for **60 seconds** (the same
TTL as the server's own policy cache — a dashboard publish reaches gated
calls within a minute). The cache holds one policy — the one the client is
currently bound to — and is validated against `realtime_policy_id` on every
call, so rebinding the client to a different policy invalidates it
immediately (alternating between two policies on one client refetches each
time; use one client per policy if you need both hot).

Failure posture, chosen deliberately:

| Situation | Behavior |
|---|---|
| Transient failure (network error, 5xx, undecodable 200) **with** a cached copy | Serve the stale copy, log a warning, and retry no sooner than the next TTL window (one fetch attempt per window during an outage — not one per call) |
| Network error / 5xx **without** any cached copy | **Raise `HTTPError`** — fail loud |
| Undecodable or malformed 200 body **without** any cached copy | **Raise `ValueError`** — fail loud |
| Unknown / unpublished / deleted policy (404), bad credentials (401) | **Raise `HTTPError`** — configuration error, never masked |

Why fail-loud with no cache: silently validating *without* the policy would
bypass governance, and silently skipping *everything* would disable
validation. Neither is acceptable for a guardrail.

---

## Discovering policies from code

Two read-only endpoints (same `X-API-Key` / `X-Project-Id` auth) expose the
project's published policies — useful for finding ids and seeing what a policy
enables before binding it:

```python
import requests

base = client.realtime_policy_base_url          # default: /realtime-validations gateway
headers = {"X-API-Key": client.api_key, "X-Project-Id": client.project_id}

# List every published policy for this project.
policies = requests.get(f"{base}/api/v1/sdk/policies", headers=headers).json()
for p in policies["data"]["policies"]:
    print(p["policy_id"], p["name"], p["enforcement"], f'v{p["version"]}')

# Inspect one policy: enabled validators, thresholds, required input fields.
detail = requests.get(f"{base}/api/v1/sdk/policies/{policy_id}", headers=headers).json()
print(detail["data"]["required_input_fields"])   # e.g. ["llm_input_query"]
for rs in detail["data"]["rulesets"]:
    for v in rs["validators"]:
        print(v["validator"], v["validator_type"], v["enabled"], v["threshold"])
```

This is the same endpoint the gated `validate()` uses internally, so what you
see here is exactly what the gate will enforce.

---

## Agentic SDK: policy on every span

Tracing an agent with `disseqt_agentic_sdk`? Attach a policy and every span
carries it as the `policy.id` resource attribute — the backend evaluates the
spans against the policy out-of-band and results land on the **Decisions**
dashboard.

```python
from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
from disseqt_agentic_sdk.enums import SpanKind

client = DisseqtAgenticClient(
    api_key="…",
    project_id="…",
    service_name="my-agent-app",              # the app name on the Decisions ledger
    realtime_policy_id="994ad00e-…",          # stamped on every span
)

with start_trace(client, "handle_request") as trace:
    with trace.start_span("llm_call", SpanKind.MODEL_EXEC) as span:
        span.set_model_info("claude-sonnet-5", "anthropic")
        ...

client.shutdown()
```

Per-trace override — different agents, different policies, one client:

```python
with start_trace(client, "risk_agent_run", realtime_policy_id="1268faa4-…") as trace:
    ...
```

The transport groups buffered spans by effective policy id and emits one POST
per distinct policy. Omit `realtime_policy_id` entirely and spans flow exactly
as before (no `policy.id` attribute).

---

## `evaluate_policy()` — deprecated

`Client.evaluate_policy(...)` — the original entry point that runs the **whole
policy server-side** and returns an aggregate BLOCK/PASS verdict — is
**deprecated in favor of policy-gated `validate()`**. Calling it emits a
`DeprecationWarning`. It stays fully functional and will not be removed
before 1.0, because it still does two things the per-validator gate does not:

1. **Aggregate verdict** — one BLOCK/PASS across all the policy's validators
   (with `is_blocking` / `is_async` / `parse_policy` helpers).
2. **Decisions-ledger entry** — a full-policy evaluation is recorded on the
   dashboard's Decisions page; individual gated `validate()` calls are regular
   validations and do not create policy decisions.

If you need either of those from the validation SDK today, keep using it:

```python
result = client.evaluate_policy(prompt=user_input)   # DeprecationWarning
if is_blocking(result):
    ...
```

Async policies (`enforcement: "async"`) also remain an `evaluate_policy`
concern: the server answers HTTP 202 (`data.status: "accepted"`) and the
verdict lands on the dashboard. Gated `validate()` runs individual validators
synchronously regardless of the policy's enforcement mode.

---

## Error handling

```python
from disseqt_sdk.client import HTTPError

try:
    result = client.validate(validator)
except ValueError:
    # Client-side: bad construction (policy without application_name) or a
    # malformed response body. Fix the call/config — don't retry.
    raise
except HTTPError as e:
    if e.status_code == 404:
        # Bound policy is unknown, unpublished, or deleted (DSQ-4040).
        alert_ops(f"bound policy not found: {e.response_body}")
    elif e.status_code == 401:
        raise                    # bad credentials
    elif e.status_code == 429:
        backoff_and_retry()      # rate limited — honor Retry-After
    else:
        backoff_and_retry()      # transient 5xx
```

Guarantees:

- A policy-fetch failure only raises when there is **no cached copy**;
  otherwise the stale copy keeps calls flowing (logged as
  `policy_gate.*_serving_stale`).
- Skip markers are **returns, not exceptions** — a validator missing from the
  policy is a normal, expected outcome.
- `HTTPError` exposes `status_code`, `message`, and `response_body`
  (truncated; never contains internal service detail).

---

## Production patterns

### Treat the policy id as configuration

```python
client = Client(
    project_id=os.environ["DISSEQT_PROJECT_ID"],
    api_key=os.environ["DISSEQT_API_KEY"],
    realtime_policy_id=os.environ.get("DISSEQT_POLICY_ID"),   # unset -> ungated
    application_name=os.environ.get("SERVICE_NAME", "my-app"),
)
```

Unset the env var and the same code runs ungated — useful for local dev.

### Handle skips explicitly at call sites that must not silently pass

A skip means "the dashboard chose not to run this check." For hard gates,
decide what a skip means for you:

```python
result = client.validate(guard_validator)
if is_policy_skipped(result):
    # Policy owner disabled this check — allow, but leave an audit trail.
    log.warning("guard %s skipped by policy %s",
                result["validator_name"], result["policy"]["policy_id"])
elif result.get("threshold_validated_result") == "Fail":
    reject()
```

### One client per policy binding

Bind the policy at construction and keep it for the client's lifetime.
Rebinding `client.realtime_policy_id` mid-flight is safe (the cached
definition is validated against the bound id on every call), but the cache
holds only one policy — distinct long-lived clients per policy are both
faster and easier to reason about.

### Latency budget

The first gated call fetches the policy (~one round-trip); every call within
the next 60 s uses the cache. Skips are pure-local (microseconds). Enabled
validators cost the same as ungated `validate()` calls.

---

## Configuration reference

`Client(...)` parameters relevant to realtime policies:

| Parameter | Default | Notes |
|---|---|---|
| `project_id` | — (required) | Sent as `X-Project-Id`. |
| `api_key` | — (required) | Sent as `X-API-Key`. |
| `realtime_policy_id` | `None` | Binds the policy that governs `validate()`. Setting it **requires** `application_name`. Unset → no gating. |
| `application_name` | `None` | Required with a policy id; identifies the app on the dashboard. |
| `realtime_policy_base_url` | `https://api.disseqt.ai/realtime-validations` | Base URL for the policy detail/discovery (and deprecated evaluate) endpoints — served by production-monitoring next to the validators. Override for local testing. |
| `timeout` | `30` | Seconds; applies to the policy fetch too. |

Public helpers: `is_policy_skipped(result)` for the gate;
`is_blocking(result)`, `is_async(result)`, `parse_policy(result)` for
(deprecated) full-policy verdicts.
