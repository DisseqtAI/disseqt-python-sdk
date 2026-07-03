# Realtime Policies

A **realtime policy** is a named, versioned bundle of validators — with their
thresholds, labels, and a decision strategy — that you author once in the
Disseqt dashboard and evaluate from your application by **policy id**, straight
from `client.validate()`:

```python
result = client.validate(
    InputValidationRequest(prompt=user_input),
    policies=["994ad00e-bff4-4cff-8a45-2635f3f3fcd0"],
)
if any_blocking(result):
    ...  # at least one policy said BLOCK
```

For each policy id, the server resolves the latest published version, runs
**every validator the policy specifies** (with the policy's thresholds), applies
the policy's decision strategy, returns one **BLOCK/PASS** verdict with a
per-rule breakdown, and records the decision on the dashboard's **Decisions**
ledger. Change a threshold or swap validators in the dashboard, publish, and
every caller picks it up — no code deploy.

---

## Table of contents

- [Quick start](#quick-start)
- [Prerequisites](#prerequisites)
- [The three call shapes](#the-three-call-shapes)
- [The response envelope](#the-response-envelope)
- [Reading policy verdicts](#reading-policy-verdicts)
- [Decision strategies](#decision-strategies)
- [Inputs: one bag, every validator](#inputs-one-bag-every-validator)
- [Sync vs. async policies](#sync-vs-async-policies)
- [Discovering policies from code](#discovering-policies-from-code)
- [Agentic SDK: policy on every span](#agentic-sdk-policy-on-every-span)
- [`evaluate_policy()` — deprecated](#evaluate_policy--deprecated)
- [Error handling](#error-handling)
- [Production patterns](#production-patterns)
- [Configuration reference](#configuration-reference)

---

## Quick start

```python
from disseqt_sdk import Client, any_blocking
from disseqt_sdk.models.input_validation import InputValidationRequest

client = Client(
    project_id="your_project_id",
    api_key="your_api_key",
    application_name="checkout-bot",   # required to evaluate policies
)

result = client.validate(
    InputValidationRequest(prompt=user_input),
    policies=["994ad00e-bff4-4cff-8a45-2635f3f3fcd0"],
)

if any_blocking(result):
    raise ValueError("Blocked by realtime policy")
```

No validator, no config, no threshold — the policy owns all of that. The
input object carries the data; the policy list says what judges it.

---

## Prerequisites

1. **A published policy.** Dashboard → **Realtime Policies → Policies →
   Create policy** (templates: Prompt-Injection Defense, PII / Data-Leakage
   Guard, …) → **Publish**. Copy the **Policy ID** (a UUID). Only *published*
   policies are visible to the SDK; the latest published version is always
   the one evaluated.
2. **A project API key** (`X-API-Key`) and **project id** (`X-Project-Id`).
3. **`application_name`** on the `Client` — required for policy evaluation;
   it's how the Decisions ledger attributes each decision to your app.

---

## The three call shapes

`validate()` accepts an optional `policies=[...]` list. What you pass as the
request decides the shape:

### 1. Validator only — classic, unchanged

```python
client.validate(
    ToxicityValidator(
        data=InputValidationRequest(prompt=user_input),
        config=SDKConfigInput(threshold=0.5),
    )
)
```

Runs that one validator with your code's config. Returns the plain
validation response — byte-identical behavior to previous releases.

### 2. Validator + policies — both, in one call

```python
result = client.validate(
    ToxicityValidator(data=InputValidationRequest(prompt=user_input),
                      config=SDKConfigInput(threshold=0.5)),
    policies=["994ad00e-…", "1268faa4-…"],
)

result["validation"]   # the toxicity result (your code's config applies to it)
result["policies"]     # one full-policy verdict per id, in order
```

The validator runs exactly as in shape 1, **and** the same input is evaluated
against each policy server-side. Useful when one ad-hoc check and the
governed policies should both see the input.

### 3. Policies only — bare request object

```python
result = client.validate(
    InputValidationRequest(prompt=user_input, response=model_output),
    policies=["994ad00e-…"],
)

result["validation"]   # None — no validator ran
result["policies"][0]  # the policy verdict
```

Any `disseqt_sdk.models` request object works as the input carrier
(`InputValidationRequest`, `OutputValidationRequest`, `RagGroundingRequest`,
`AgenticBehaviourRequest`, `McpSecurityRequest`) — pick whichever names the
fields you have. No validator, no config.

**Rules enforced client-side** (all raise `ValueError` before any network
call): a bare request without `policies`; an empty `policies` list or blank
ids; `policies` combined with composite-score or themes-classifier requests;
missing `application_name`; a request that serializes to no input fields.

---

## The response envelope

Whenever `policies` is passed, the return value is a stable two-key envelope:

```python
{
    "validation": {...} | None,   # per-validator result (None in shape 3)
    "policies":  [{...}, ...],    # one policy envelope per id, same order
}
```

Each entry in `"policies"` is the standard policy envelope:

```jsonc
{
  "status": "success",
  "code": "DSQ-2000",              // DSQ-2020 for an async 202
  "request_id": "…",
  "data": {
    "policy_id": "…",
    "policy_name": "…",
    "policy_version": 3,
    "status": "completed",         // "accepted" for async
    "decision": "BLOCK",           // omitted on async
    "enforcement": "sync",
    "rulesets": [ /* per-rule breakdown; omitted on async */ ],
    "duration": "…",
    "credit_details": { /* sync only */ }
  }
}
```

Policies are evaluated **sequentially, in the order given**. Each policy is
one server-side evaluation: billed per executed validator, one
Decisions-ledger entry. Keep the list short (1–3 is typical: an org-wide
baseline plus an app policy).

---

## Reading policy verdicts

```python
from disseqt_sdk import any_blocking, is_blocking, is_async, parse_policy

result = client.validate(req, policies=[P1, P2])

# The one-line gate:
if any_blocking(result):          # True if ANY policy said BLOCK
    reject()

# Per-policy:
for envelope in result["policies"]:
    decision = parse_policy(envelope)      # typed PolicyDecision | None
    print(decision.policy_name, decision.decision, decision.enforcement)
    for ruleset in decision.rulesets:
        for rule in ruleset.rules:
            print(" ", rule.validator, rule.status, rule.score, rule.threshold)
```

`any_blocking` accepts the whole envelope, a list of policy envelopes, or a
single envelope — and returns `False` for anything else (including a classic
validator response), so it's always safe to gate on. `is_blocking` /
`is_async` / `parse_policy` work on individual envelopes as before.

---

## Decision strategies

The policy's **aggregation strategy** (set in the dashboard's Enforcement
tab) decides how per-rule outcomes combine into the BLOCK/PASS verdict:

| Strategy | Verdict |
|---|---|
| `any` (default) | BLOCK if **any** rule failed |
| `all` | "All must pass" — BLOCK if any executed rule failed **or errored** |
| `majority` | BLOCK if failed rules are a strict majority of the pass/fail votes; ties PASS |
| `weighted` | BLOCK when the **policy confidence ≥ threshold**. Confidence is the ruleset-weight-weighted mean of per-ruleset badness (risk validators contribute their score, quality validators `1 − score`, averaged over the ruleset's scored rules) |

Under every strategy, **skipped rules stay neutral**: they cast no vote, and
under `weighted` a fully-skipped ruleset's weight is *renormalized away* —
the confidence is computed over what actually ran, with the configured
relative weights intact. A policy whose rules all skipped still passes by
vacuity. Two things override every strategy: an application-level
`overrides_block` forces BLOCK, and an error on an `is_decider` rule forces
BLOCK.

The decision explains itself in the envelope and on `PolicyDecision`:

```python
d = parse_policy(envelope)
d.aggregation           # "any" | "all" | "majority" | "weighted"
d.aggregate_score       # weighted only: policy confidence in [0, 1]
d.aggregate_threshold   # weighted only: the blocking line (score >= thr -> BLOCK)
```

`aggregate_score` is `None` for non-weighted strategies and for vacuous
weighted decisions (nothing scored); `aggregation` is empty on servers that
predate enforcement.

---

## Inputs: one bag, every validator

A policy can span multiple validator domains. The request object's fields are
serialized once and sent to **every** validator the policy lists — each reads
the fields it needs:

| Request field | Wire field | Read by |
|---|---|---|
| `prompt` | `llm_input_query` | input / RAG / security validators |
| `context` | `llm_input_context` | RAG / output validators |
| `response` | `llm_output` | output / RAG / security validators |
| `conversation_history`, `tool_calls`, `agent_responses`, `reference_data` | same names | agentic validators |

If a validator's required field is missing, the server records that rule as
`status: "skipped"` with `skipped_reason: "missing_input:<fields>"` — skips
are **neutral** to the decision, never billed, and name exactly what to add.
Supply the union of what your policies need; the
[discovery endpoint](#discovering-policies-from-code) tells you the union
up front via `required_input_fields`.

---

## Sync vs. async policies

A policy's **enforcement** (execution mode, set at authoring time) decides
what its envelope contains:

- **Sync** — validators run in-line; HTTP 200, `data.status: "completed"`,
  `decision` + `rulesets` present. This is the guardrail case.
- **Async** — the request is accepted (HTTP 202, `code: "DSQ-2020"`,
  `data.status: "accepted"`), evaluation runs in the background, and the
  verdict lands on the **Decisions** dashboard. The envelope carries **no
  decision and no rulesets** — `any_blocking` treats it as not blocking.
  Use `is_async(envelope)` to detect it, and the envelope's `request_id`
  to reconcile the eventual dashboard decision.

Mixing sync and async policies in one `policies=[...]` list is fine — each
envelope self-describes.

---

## Discovering policies from code

Two read-only endpoints (same auth headers) expose the project's published
policies:

```python
import requests

base = client.realtime_policy_base_url
headers = {"X-API-Key": client.api_key, "X-Project-Id": client.project_id}

# All published policies:
listing = requests.get(f"{base}/api/v1/sdk/policies", headers=headers).json()
for p in listing["data"]["policies"]:
    print(p["policy_id"], p["name"], p["enforcement"], f'v{p["version"]}')

# One policy — including the inputs it needs:
detail = requests.get(f"{base}/api/v1/sdk/policies/{policy_id}", headers=headers).json()
print(detail["data"]["required_input_fields"])   # e.g. ["llm_input_query"]
```

`required_input_fields` is the union over the policy's **enabled** validators
— build your request object from it and nothing will skip.

---

## Agentic SDK: policy on every span

Tracing an agent with `disseqt_agentic_sdk`? Attach a policy and every span
carries it as the `policy.id` resource attribute; evaluation happens
out-of-band and results land on the Decisions dashboard.

```python
from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
from disseqt_agentic_sdk.enums import SpanKind

client = DisseqtAgenticClient(
    api_key="…", project_id="…",
    service_name="my-agent-app",          # the app name on the ledger
    realtime_policy_id="994ad00e-…",      # stamped on every span
)

with start_trace(client, "handle_request") as trace:
    with trace.start_span("llm_call", SpanKind.MODEL_EXEC) as span:
        span.set_model_info("claude-sonnet-5", "anthropic")
        ...

client.shutdown()
```

Per-trace override: `start_trace(client, "risk_agent", realtime_policy_id="1268faa4-…")`.
The transport groups spans by effective policy id — one POST per distinct
policy. Omit the policy id entirely and spans flow exactly as before.

---

## `evaluate_policy()` — deprecated

`Client.evaluate_policy(...)` predates `validate(..., policies=[...])` and
hits the same endpoint. It is deprecated (calling it emits a
`DeprecationWarning`) but stays fully functional until 1.0 — it still offers
typed kwargs (`prompt=`, `context=`, …), `config_input`, and an explicit
`request_id` override. New code should use `validate(...)` with `policies`.

---

## Error handling

```python
from disseqt_sdk.client import HTTPError

try:
    result = client.validate(req, policies=[P1])
except ValueError:
    # Client-side: invalid combination (see the rules under "The three
    # call shapes") or an undecodable response body. Fix the call.
    raise
except HTTPError as e:
    if e.status_code == 404:
        # Unknown, unpublished, or deleted policy id (DSQ-4040).
        alert_ops(f"policy not found: {e.response_body}")
    elif e.status_code == 401:
        raise                    # bad credentials
    elif e.status_code == 429:
        backoff_and_retry()      # rate limited — honor Retry-After
    else:
        backoff_and_retry()      # transient 5xx
```

Facts to rely on:

- **Unknown / unpublished / deleted policy → HTTP 404** (`DSQ-4040`); a
  malformed (non-UUID) id also answers 404 on current servers.
- **Sequential semantics on failure**: policies evaluate in order; if policy
  N fails, policies 1..N-1 already ran (and were recorded server-side), the
  validator (shape 2) already ran, and the exception propagates. Treat a
  raised `HTTPError` as "gate undecided" and apply your fail-open/closed
  stance.
- Error bodies never leak internal service detail.

---

## Production patterns

### Policy ids are configuration

```python
POLICIES = [p for p in os.environ.get("DISSEQT_POLICIES", "").split(",") if p]

result = client.validate(req, policies=POLICIES) if POLICIES else client.validate(validator)
```

Empty env → ungated code path; populated → governed. No deploy to change
which policies apply.

### Fail open vs. fail closed

```python
def is_allowed(user_input: str) -> bool:
    try:
        result = client.validate(
            InputValidationRequest(prompt=user_input), policies=POLICIES
        )
    except HTTPError as e:
        if e.status_code in (401, 404):
            raise                 # misconfiguration — surface loudly
        log.error("policy evaluation failed; failing closed", exc_info=True)
        return False              # guardrail: an outage must not open the gate
    return not any_blocking(result)
```

### Correlate with `request_id`

Each policy envelope carries a server-generated `request_id` — log it; for
async policies it's the handle that matches the eventual Decisions-ledger
entry.

### Rate limits and cost

Endpoints are rate-limited per API key (`429` + `Retry-After`). Each policy
in the list is a separate evaluation billed per executed validator — N
policies ≈ N× the cost of one. Prefer one well-composed policy over many
overlapping ones.

---

## Configuration reference

`Client(...)` parameters relevant to realtime policies:

| Parameter | Default | Notes |
|---|---|---|
| `project_id` | — (required) | Sent as `X-Project-Id`. |
| `api_key` | — (required) | Sent as `X-API-Key`. |
| `application_name` | `None` | **Required** for `policies=[...]` (and for `realtime_policy_id`); shown on the Decisions ledger. |
| `realtime_policy_id` | `None` | Default policy for the deprecated `evaluate_policy()` only — `validate()` takes explicit `policies=[...]`. |
| `realtime_policy_base_url` | `https://api.disseqt.ai/realtime-validations` | Base URL for the policy evaluate + discovery endpoints (served by production-monitoring next to the validators). Override for local testing. |
| `timeout` | `30` | Seconds, per HTTP call (each policy evaluation is one call). |

Public helpers: `any_blocking(result)` for the envelope; `is_blocking` /
`is_async` / `parse_policy` for individual policy envelopes.
