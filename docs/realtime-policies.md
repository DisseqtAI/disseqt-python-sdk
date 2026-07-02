# Realtime Policies

A **realtime policy** is a named, versioned bundle of validators — with their
thresholds, labels, and an enforcement strategy — that you author once in the
Disseqt dashboard and then run from your application by **policy id**. Instead
of wiring up individual validators in code, you call one method and get back a
single **BLOCK / PASS** decision plus the full per-rule breakdown.

Use realtime policies when you want the *policy* (which validators, what
thresholds, block-vs-monitor) to live in the dashboard and change without a code
deploy — the SDK just references it by id.

> **How this differs from `client.validate(...)`**
> `validate()` runs one validator (or a fixed composite bundle) that you
> configure in code. `evaluate_policy()` runs a *published policy* that the
> server resolves by id — the set of validators and thresholds is owned by the
> dashboard, not your code. See [Choosing an approach](#choosing-an-approach).

---

## Table of contents

- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [Two SDKs, two entry points](#two-sdks-two-entry-points)
- [`evaluate_policy` — the request](#evaluate_policy--the-request)
- [Reading the result](#reading-the-result)
- [Sync vs. async policies](#sync-vs-async-policies)
- [Discovering policies from code](#discovering-policies-from-code)
- [Agentic SDK: policy on every span](#agentic-sdk-policy-on-every-span)
- [Error handling](#error-handling)
- [Production patterns](#production-patterns)
- [Configuration reference](#configuration-reference)
- [Choosing an approach](#choosing-an-approach)

---

## Quick start

```python
from disseqt_sdk import Client, is_blocking

client = Client(
    project_id="your_project_id",
    api_key="your_api_key",
    realtime_policy_id="994ad00e-bff4-4cff-8a45-2635f3f3fcd0",  # published policy
    application_name="checkout-bot",                            # who is calling
)

result = client.evaluate_policy(prompt=user_input)

if is_blocking(result):
    # Policy said BLOCK — do not send this to the model / downstream.
    raise ValueError("Request blocked by realtime policy")
```

That is the whole happy path: point the client at a policy id, pass the input,
and branch on `is_blocking`.

---

## Prerequisites

1. **A published policy.** Create one in the dashboard under
   **Realtime Policies → Policies** (start from a template or build from
   scratch), then **Publish**. Copy the **Policy ID** shown on publish — that
   UUID is what the SDK references. A policy only affects traffic once it is
   *published*; drafts are invisible to the SDK.
2. **A project API key** (`X-API-Key`) and **project id** (`X-Project-Id`) — the
   same credentials you use for `client.validate(...)`.
3. **`application_name`.** Required whenever a policy id is set. It is recorded
   on every decision so the dashboard's **Decisions** ledger can show which
   application produced each verdict. Think of it as the caller's logical name
   (`"checkout-bot"`, `"support-agent"`), not a hostname.

---

## Two SDKs, two entry points

Realtime policies are reachable from both packages:

| Package | Entry point | Use when |
|---|---|---|
| `disseqt_sdk` | `Client.evaluate_policy(...)` | You want a **synchronous verdict in-line** — a request/response guardrail you branch on immediately. |
| `disseqt_agentic_sdk` | `DisseqtAgenticClient(realtime_policy_id=...)` / `start_trace(..., realtime_policy_id=...)` | You are **tracing an agent** and want each span evaluated against a policy out-of-band, with results on the dashboard. |

The two are independent — pick by whether you need the decision back in the call
(`evaluate_policy`) or you are emitting spans and want them policy-tagged
(agentic). Both are covered below.

---

## `evaluate_policy` — the request

```python
result = client.evaluate_policy(
    realtime_policy_id="…",     # optional if set on the Client; per-call wins
    prompt="…",                 # user query          → wire: llm_input_query
    context="…",                # retrieved context   → wire: llm_input_context
    response="…",               # model output        → wire: llm_output
    # agentic inputs (sent as-is, for policies that include agentic validators)
    conversation_history=[...],
    tool_calls=[...],
    agent_responses=[...],
    reference_data={...},
    # escape hatch + overrides
    input_data={...},           # raw dict, merged last (raw keys win on conflict)
    config_input={...},         # extra validator config (policy threshold wins)
    application_name="…",       # optional per-call override of the Client default
    request_id="…",             # optional; server generates one if omitted
)
```

### Typed inputs and the field rename

The typed keyword arguments are renamed on the wire to the shape the validators
expect — the same convention as `InputValidationRequest` /
`OutputValidationRequest`:

| Keyword arg | Wire field | For |
|---|---|---|
| `prompt` | `llm_input_query` | the user's query |
| `context` | `llm_input_context` | retrieved / supplied context |
| `response` | `llm_output` | the model's output to check |
| `conversation_history` | `conversation_history` | agentic validators (turn history) |
| `tool_calls` | `tool_calls` | agentic validators (tool invocations) |
| `agent_responses` | `agent_responses` | agentic validators |
| `reference_data` | `reference_data` | agentic validators (ground truth) |

A single policy can span multiple validator domains. The **same** input is sent
to every validator the policy lists, and each validator reads the fields it
needs — so supply the **union** of what the policy's validators require:

```python
result = client.evaluate_policy(
    # LLM validators read these
    prompt="What is the capital of France?",
    context="France is a country in Europe.",
    response="The capital of France is Paris.",
    # agentic validators read these
    tool_calls=[{"name": "lookup_capital", "args": {"country": "France"}}],
)
```

If a validator's required field is missing, the server does **not** guess — it
records that rule as `status="skipped"` with
`skipped_reason="missing_input:<fields>"` so you can see exactly what to add.
[Discover a policy's required fields](#discovering-policies-from-code) up front
to avoid this.

### The raw escape hatch

For shapes the typed args don't cover (e.g. a themes classifier, or a custom
validator), pass `input_data` as a raw dict. It is merged **last**, so raw keys
win on conflict:

```python
result = client.evaluate_policy(input_data={"llm_input_query": "…", "custom_field": "…"})
```

### `config_input` and the threshold rule

`config_input` fills in validator config the policy didn't set. **Any key the
policy already defines always wins** — for a validator the policy configures
(its threshold, custom labels, label scores, …), a conflicting `config_input`
key is ignored; `config_input` only supplies keys the policy left unset. This is
deliberate: a caller cannot weaken dashboard-enforced guardrails — e.g. lower a
threshold — from the client side.

---

## Reading the result

`evaluate_policy` returns the decoded JSON response — the standard Disseqt
envelope with the verdict under `data`. **Prefer the helper functions** over
indexing the raw dict; they unwrap the envelope for you and are stable across
minor response-shape changes.

```python
from disseqt_sdk import is_blocking, is_async, parse_policy

result = client.evaluate_policy(prompt=user_input)

# 1. Short-circuit on BLOCK — the common case.
if is_blocking(result):
    handle_block()

# 2. Full structured breakdown when you need per-rule detail.
decision = parse_policy(result)          # -> PolicyDecision | None
print(decision.decision)                 # "BLOCK" | "PASS"
print(decision.enforcement)              # "sync" | "async"
print(decision.policy_name, "v", decision.policy_version)

for ruleset in decision.rulesets:
    for rule in ruleset.rules:
        print(rule.validator, rule.status, rule.score, rule.threshold)
```

### Typed result objects

`parse_policy()` returns a `PolicyDecision` (or `None` if the payload isn't a
policy result). All three types are **frozen** (immutable) and slotted
(`from dataclasses import dataclass, field`):

```python
@dataclass(frozen=True, slots=True)
class PolicyDecision:
    policy_id: str
    policy_name: str
    policy_version: int
    decision: str                          # "BLOCK" | "PASS"
    enforcement: str                       # "sync" | "async"
    rulesets: list[PolicyRuleset] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class PolicyRuleset:
    ruleset_id: str
    ruleset_name: str
    required: bool = False
    rules: list[PolicyRule] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class PolicyRule:
    validator: str                         # e.g. "prompt-injection"
    validator_type: str                    # e.g. "mcp-security"
    status: str                            # "pass" | "fail" | "skipped" | "error"
    score: float | None = None             # None when the validator produced no score
    threshold: float | None = None
    polarity: str = ""                     # "risk" (higher = worse) | "quality" (higher = better)
    is_decider: bool = False
    skipped_reason: str = ""               # e.g. "missing_input:llm_input_query"
```

Because they are frozen, treat results as read-only — assigning to a field
(e.g. `decision.decision = "PASS"`) raises `FrozenInstanceError`.

> **On `is_decider`.** This flag is a *policy-authoring* setting, not "the rule
> that caused this block." Under an **any-fails → block** strategy, *any*
> failing validator blocks regardless of `is_decider`; the flag only controls
> whether a validator **error** (e.g. an ML service failure) is enough to flip
> the verdict to BLOCK. Don't infer the deciding rule from `is_decider` — read
> `status` and the policy's strategy instead.

### Response envelope (for reference)

If you must read the raw dict, this is the shape:

```jsonc
{
  "status": "success",
  "code": "DSQ-2000",              // DSQ-2020 on the async 202
  "request_id": "…",               // envelope-level correlation id
  "timestamp": "…",
  "data": {
    "policy_id": "…",
    "policy_name": "…",
    "policy_version": 3,
    "status": "completed",         // "accepted" for async
    "decision": "BLOCK",           // omitted on async
    "enforcement": "sync",         // "sync" | "async"
    "rulesets": [ /* omitted on async */ ],
    "duration": "…",
    "credit_details": { /* sync only */ }
  }
}
```

---

## Sync vs. async policies

A policy's **enforcement** (its `strategy.executionMode`) determines how the
call behaves. It is **decoupled** from the BLOCK/PASS verdict.

### Sync (`enforcement: "sync"`)

The server runs every validator in-line and returns the full verdict — HTTP 200,
`data.status="completed"`, `decision` and `rulesets` populated. This is the
guardrail case: you get the decision back and branch on it.

```python
result = client.evaluate_policy(prompt=user_input)
if is_blocking(result):
    reject()
```

### Async (`enforcement: "async"`)

The server **accepts** the request (HTTP **202**, `code="DSQ-2020"`,
`data.status="accepted"`), runs the evaluation in the background, and publishes
the verdict to the **Decisions** dashboard. The 202 response carries **no
`decision` and no `rulesets`** — there is nothing to branch on synchronously.

```python
result = client.evaluate_policy(response=model_output)

if is_async(result):
    # Fire-and-forget: the verdict will appear on the dashboard, not here.
    log.info("submitted for async evaluation", extra={
        "request_id": result.get("request_id"),
    })
```

Use async for monitoring/observability where you don't need to gate the response
in real time. Use sync when the decision must block the request. **`is_blocking`
safely returns `False` on an async 202** (there is no decision yet), so a guard
written for sync won't accidentally block on an async policy — but you should
choose the enforcement mode deliberately per use case.

---

## Discovering policies from code

You don't have to hard-code policy ids or guess which inputs a policy needs. Two
read-only, API-key-authenticated endpoints let you discover both. They are not
yet wrapped as SDK methods, so call them directly (they share the client's base
URL and headers):

```python
import requests

base = client.realtime_policy_base_url          # default: /realtime-validations gateway
headers = {"X-API-Key": client.api_key, "X-Project-Id": client.project_id}

# List every published policy for this project.
policies = requests.get(f"{base}/api/v1/sdk/policies", headers=headers).json()
for p in policies["data"]["policies"]:
    print(p["policy_id"], p["name"], p["enforcement"], f'v{p["version"]}')

# Inspect one policy — including the exact input fields it needs.
detail = requests.get(f"{base}/api/v1/sdk/policies/{policy_id}", headers=headers).json()
print("required inputs:", detail["data"]["required_input_fields"])
# e.g. ["llm_input_query"] — pass these to evaluate_policy to avoid skips.
```

`required_input_fields` is the union of what the policy's **enabled** validators
require, so you can assemble a correct `evaluate_policy(...)` call up front
rather than discovering `skipped_reason` at runtime.

---

## Agentic SDK: policy on every span

When you are tracing an agent with `disseqt_agentic_sdk`, attach a policy id and
every span carries it as the `policy.id` resource attribute. The backend reads
that attribute and evaluates the spans against the policy — results land on the
dashboard.

**Client default — one policy for the whole application:**

```python
from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
from disseqt_agentic_sdk.enums import SpanKind

client = DisseqtAgenticClient(
    api_key="…",
    project_id="…",
    service_name="my-agent-app",                 # also the app name on the dashboard
    realtime_policy_id="994ad00e-…",             # stamped on every span
)

with start_trace(client, "handle_user_request") as trace:
    with trace.start_span("llm_call", SpanKind.MODEL_EXEC) as span:
        span.set_model_info("claude-sonnet-5", "anthropic")
        ...

client.shutdown()
```

**Per-trace override — different policies for different agents in one app:**

```python
# Client has a default policy; this trace runs under a different one.
with start_trace(client, "risk_agent_run", realtime_policy_id="1268faa4-…") as trace:
    ...
```

The transport groups buffered spans by effective policy id and emits **one POST
per distinct policy**, so a client-default policy and a per-trace override are
delivered correctly in the same session. `service_name` is required on the
agentic client and doubles as the application name on the dashboard (the
agentic equivalent of `application_name`).

---

## Error handling

`evaluate_policy` raises two exception types. Handle them distinctly:

```python
from disseqt_sdk.client import HTTPError

try:
    result = client.evaluate_policy(prompt=user_input)
except ValueError as e:
    # Client-side guard: no policy id, no application_name, or no input fields.
    # These are programming errors — fix the call, don't retry.
    raise
except HTTPError as e:
    if e.status_code == 404:
        # Unknown, unpublished, or deleted policy id (DSQ-4040).
        # Also returned for a malformed (non-UUID) id.
        alert_ops(f"policy not found: {e.response_body}")
    elif e.status_code == 401:
        # Bad or missing API key / project id.
        raise
    elif e.status_code == 429:
        # Rate limited — back off and retry (see Retry-After on the response).
        backoff_and_retry()
    else:
        # 5xx — transient server/upstream error. Retry with backoff.
        backoff_and_retry()
```

**Guarantees worth relying on:**

- **Unknown / unpublished / deleted policy → HTTP 404** (`DSQ-4040`). Branch on
  `e.status_code == 404` to tell "bad policy id" apart from a server fault.
  *(Deployments older than production-monitoring v0.1.12 returned 500 for these;
  a malformed non-UUID id may still surface as 500 on servers without the
  realtime-policies-service malformed-id fix.)*
- **Error bodies never leak internal detail** — upstream URLs and stack detail
  are kept server-side; the `external` message is caller-safe.
- **`ValueError` is always client-side** — it means the call itself was
  malformed (missing policy id / `application_name` / input). No network request
  was made.

---

## Production patterns

### Set the policy once, evaluate many times

Configure the policy id and application name on the `Client`; omit them per call.
A per-call `realtime_policy_id` always overrides the default when you need it.

```python
client = Client(
    project_id=PROJECT_ID,
    api_key=API_KEY,
    realtime_policy_id=POLICY_ID,
    application_name="checkout-bot",
    timeout=30,                       # seconds; raise for slow multi-validator policies
)

# Every call inherits the default policy + application_name.
verdict = client.evaluate_policy(prompt=user_input)
```

### Fail open vs. fail closed

Decide explicitly what happens when evaluation itself fails (network, 5xx). For a
hard guardrail, **fail closed** (treat an error as BLOCK); for advisory
monitoring, **fail open**. Make it a conscious choice, not an accident of
exception flow:

```python
def is_allowed(user_input: str) -> bool:
    try:
        return not is_blocking(client.evaluate_policy(prompt=user_input))
    except HTTPError as e:
        if e.status_code == 404:
            raise                      # misconfiguration — surface loudly
        # Guardrail policy: an evaluation outage should not silently open the gate.
        log.error("policy eval failed; failing closed", exc_info=True)
        return False                   # fail closed
```

### Correlate calls with `request_id`

Pass your own `request_id` to tie a policy call to your request logs; the server
echoes it back in the envelope. If you omit it, the server generates one — read
it from `result["request_id"]`. For **async** policies this id is your only
handle to reconcile the 202 with the verdict that later appears on the dashboard.

```python
result = client.evaluate_policy(prompt=user_input, request_id=trace_id)
assert result["request_id"] == trace_id
```

### Rate limits

The SDK policy endpoints are rate-limited per API key. On a `429`, the response
carries `Retry-After` / `X-RateLimit-*` headers — honor them with backoff. High-
throughput callers should batch or spread load rather than bursting.

### Credits

Each **sync** evaluation costs one credit regardless of how many validators the
policy runs; `result["data"]["credit_details"]` reports the deduction. Async 202
responses do not carry `credit_details` (billing settles when the background
evaluation completes).

---

## Configuration reference

`Client(...)` parameters relevant to realtime policies:

| Parameter | Default | Notes |
|---|---|---|
| `project_id` | — (required) | Sent as `X-Project-Id`. |
| `api_key` | — (required) | Sent as `X-API-Key`. |
| `realtime_policy_id` | `None` | Default policy for `evaluate_policy`. Per-call value wins. Setting it **requires** `application_name`. |
| `application_name` | `None` | Required when a policy id is set; recorded on every decision. |
| `realtime_policy_base_url` | `https://api.disseqt.ai/realtime-validations` | Base URL for the evaluate + discovery endpoints. The evaluate endpoint is served by production-monitoring next to the validators — **not** the `/realtime-policies` dashboard gateway. Override for local testing (e.g. `http://localhost:9010`). |
| `timeout` | `30` | Request timeout in seconds. Raise for policies with many validators. |

`evaluate_policy(...)` raises `ValueError` (client-side) if you call it without a
policy id, without an `application_name`, or with no input fields — see
[Error handling](#error-handling).

---

## Choosing an approach

| You want to… | Use |
|---|---|
| Run **one validator** you configure in code | `client.validate(SomeValidator(...))` |
| Run a **fixed bundle** of validators with weighted scoring | `client.validate(CompositeScoreEvaluator(...))` |
| Run a **dashboard-managed policy** by id and get a BLOCK/PASS verdict | `client.evaluate_policy(...)` |
| **Tag agent spans** with a policy for out-of-band evaluation | `DisseqtAgenticClient(realtime_policy_id=...)` |

If the *set of validators and thresholds* should be owned by the dashboard and
changeable without a code deploy, use a realtime policy. If it lives in your
code, use `validate()`.
