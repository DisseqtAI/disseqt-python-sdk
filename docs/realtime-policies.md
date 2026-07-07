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
- [Rule statuses and skip reasons](#rule-statuses-and-skip-reasons)
- [Reading policy verdicts](#reading-policy-verdicts)
- [Decision strategies](#decision-strategies)
- [Input coverage: all, some, or none](#input-coverage-all-some-or-none)
- [Inputs: one bag, every validator](#inputs-one-bag-every-validator)
- [Validator overrides](#validator-overrides)
- [Sync vs. async policies](#sync-vs-async-policies)
- [Discovering policies from code](#discovering-policies-from-code)
- [Agentic SDK: policy on every span](#agentic-sdk-policy-on-every-span)
- [Error handling](#error-handling)
- [Billing, latency, and publish propagation](#billing-latency-and-publish-propagation)
- [The Decisions ledger](#the-decisions-ledger)
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

### Client-level default: a governed client

Set the list once on the client and every `validate()` call is policy-checked
— the per-call `policies=` acts as an **override** when present:

```python
client = Client(
    project_id=..., api_key=...,
    application_name="checkout-bot",
    policies=["994ad00e-…"],              # the governed default
)

client.validate(toxicity_validator)        # shape 2: validator + default policies
client.validate(InputValidationRequest(prompt=p))   # shape 3 via the default
client.validate(req, policies=["1268faa4-…"])       # override: only this list runs
```

Semantics worth knowing:

- **Per-call always wins** — the override replaces the default, it doesn't
  append to it. There is no per-call opt-out; construct a second `Client`
  for deliberately ungoverned paths.
- **`policies=[]` is always an error**, with or without a default — an
  accidentally empty list must fail loudly rather than silently ungate the
  call. `Client(policies=[])` at construction, by contrast, just means "no
  default", so env-driven lists degrade naturally.
- **Composite-score and themes-classifier requests** can't be policy-
  evaluated; on a governed client they run classically and the default
  steps aside (logged as `validation.policies.default_skipped`). Passing
  `policies` to them *explicitly* still raises.
- The list is **copied at construction** — mutating your original list
  later doesn't change the client.

---

## The response envelope

Whenever `policies` is passed, the return value is a stable two-key envelope:

```python
{
    "validation": {...} | None,   # per-validator result (None in shape 3)
    "policies":  [{...}, ...],    # one policy envelope per id, same order
}
```

Each entry in `"policies"` is the standard policy envelope. A real sync
verdict, annotated:

```jsonc
{
  "status": "success",
  "code": "DSQ-2000",                    // DSQ-2020 for an async 202
  "request_id": "d93onau…",              // correlate logs & ledger with this
  "data": {
    "policy_id": "994ad00e-…",
    "policy_name": "Prompt-Injection Defense",
    "policy_version": 3,                 // the published version that judged
    "status": "completed",               // "accepted" for async
    "decision": "BLOCK",                 // BLOCK | PASS — omitted on async
    "enforcement": "sync",               // sync | async
    "aggregation": "weighted",           // strategy that decided the verdict
    "aggregate_score": 0.79,             // weighted only: policy confidence
    "aggregate_threshold": 0.5,          // weighted only: score >= thr -> BLOCK
    "rulesets": [                        // per-rule breakdown — omitted on async
      {
        "ruleset_id": "rs_a1",
        "ruleset_name": "Injection",
        "required": false,
        "rules": [
          {
            "validator": "prompt-injection",
            "validator_type": "mcp-security",
            "status": "fail",            // pass | fail | skipped | error
            "score": 0.9876,
            "has_score": true,           // false => score is meaningless
            "threshold": 0.6,            // the policy's per-rule threshold
            "polarity": "risk",          // risk | quality
            "is_decider": false,
            "skipped_reason": ""
          }
        ]
      },
      {
        "ruleset_id": "rs_a2",
        "ruleset_name": "Output leakage",
        "required": false,
        "rules": [
          {
            "validator": "data-leakage",
            "validator_type": "output-validation",
            "status": "skipped",
            "score": 0,
            "has_score": false,
            "threshold": 0.55,
            "polarity": "risk",
            "is_decider": false,
            "skipped_reason": "missing_input:llm_output"
          }
        ]
      }
    ],
    "duration": "1.24s",
    "credit_details": {                  // present when validators executed;
      "credits_deducted": 2              // per executed validator — skips are
      /* …additional balance fields… */  // free. Omitted on async 202s and on
    }                                    // vacuous evaluations (nothing ran)
  }
}
```

Field notes:

- **`aggregation` / `aggregate_score` / `aggregate_threshold`** — which
  [decision strategy](#decision-strategies) produced the verdict and, for
  `weighted`, the confidence it compared against the blocking line. Empty /
  absent on servers that predate aggregation enforcement.
- **`rulesets`** mirrors the policy editor's structure — one entry per
  ruleset, in policy order, each rule carrying its own outcome. See the
  [status vocabulary](#rule-statuses-and-skip-reasons) below.
- **`has_score: false`** marks rules that produced no score — skipped rules
  and override-forced verdicts. Their `score` serializes as `0` on the wire,
  so never read `score` without checking `has_score` (the typed
  `parse_policy` does this for you and gives you `score=None`). Errored
  rules may carry **either** value (a parseable ML error can include a
  meaningless zero-valued score) — branch on `status` first and only trust
  `score` for `pass`/`fail` rules.
- **`request_id`** is stamped by the server (or taken from your
  `X-Request-Id`); the same id links the response, the server logs, and the
  Decisions-ledger evidence for this evaluation.

Policies are evaluated **sequentially, in the order given**. Each policy is
one server-side evaluation: billed per executed validator, one
Decisions-ledger entry. Keep the list short (1–3 is typical: an org-wide
baseline plus an app policy).

---

## Rule statuses and skip reasons

Every rule in the breakdown ends in exactly one of four states:

| `status` | Meaning | Effect on the decision | Billed |
|---|---|---|---|
| `pass` | Ran; score on the passing side of the rule's threshold | Counts as a pass | Yes |
| `fail` | Ran and breached its threshold — or was force-failed by a [block override](#validator-overrides) | Blocks under `any`/`all`; a fail vote under `majority`; raises the confidence under `weighted`; a forced fail blocks under **every** strategy | Yes (forced fails: no) |
| `skipped` | Never ran | **Neutral** under every strategy — casts no vote, contributes no weight | No |
| `error` | Ran, but the validator errored | Blocks if the rule is marked `is_decider` (any strategy) and under `all` (an executed non-pass); otherwise recorded but neutral. Excluded from `weighted` math | — |

`skipped_reason` says *why* a rule skipped (or why a fail was forced):

| `skipped_reason` | On status | Meaning |
|---|---|---|
| `missing_input:<field,…>` | `skipped` | The request didn't carry the wire fields this validator needs — the list names exactly what to add |
| `overrides_allow` | `skipped` | The policy's allow-override list exempts this validator |
| `overrides_block` | `fail` | The policy's block-override list force-fails this validator — no score, `has_score: false`, blocks regardless of strategy |
| `no_result` | `skipped` | The validator ran but returned no usable result (rare; treated as a skip) |

Two invariants worth building on: **skips are never billed**, and **a skip
can never flip a verdict to BLOCK** — only real fails (or decider errors, or
forced fails) block.

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

### The typed objects

`parse_policy(envelope)` accepts the full DSQ envelope **or** the unwrapped
`data` dict, and returns `None` when the payload carries no `policy_id`
(e.g. an error envelope). Otherwise:

**`PolicyDecision`**

| Field | Type | Notes |
|---|---|---|
| `policy_id`, `policy_name`, `policy_version` | `str`, `str`, `int` | The published version that judged |
| `decision` | `str` | `"BLOCK"` \| `"PASS"` |
| `enforcement` | `str` | `"sync"` \| `"async"` |
| `aggregation` | `str` | `"any"` \| `"all"` \| `"majority"` \| `"weighted"`; `""` on older servers |
| `aggregate_score` | `float \| None` | Weighted only: the policy confidence. `None` otherwise, and on vacuous decisions |
| `aggregate_threshold` | `float \| None` | Weighted only: `score >= threshold → BLOCK` |
| `rulesets` | `list[PolicyRuleset]` | Policy-editor structure, in order |

**`PolicyRuleset`**: `ruleset_id`, `ruleset_name`, `required`, `rules:
list[PolicyRule]`.

**`PolicyRule`**

| Field | Type | Notes |
|---|---|---|
| `validator` | `str` | Wire name, e.g. `"prompt-injection"` |
| `validator_type` | `str` | Domain, e.g. `"input-validation"`, `"mcp-security"` |
| `status` | `str` | `pass` \| `fail` \| `skipped` \| `error` |
| `score` | `float \| None` | **`None` whenever the wire `has_score` is false** — no fabricated zeros |
| `threshold` | `float \| None` | The policy's per-rule threshold |
| `polarity` | `str` | `"risk"` (high = bad) \| `"quality"` (high = good) |
| `is_decider` | `bool` | An `error` on this rule blocks the policy |
| `skipped_reason` | `str` | See the [vocabulary](#rule-statuses-and-skip-reasons); `""` when not applicable |

---

## Decision strategies

The policy's **aggregation strategy** (dashboard → policy editor →
Enforcement tab → *Advanced → Aggregation*) decides how per-rule outcomes
combine into the BLOCK/PASS verdict:

| Strategy | Verdict |
|---|---|
| `any` (default) | BLOCK if **any** rule failed |
| `all` | "All must pass" — BLOCK if any executed rule failed **or errored** |
| `majority` | BLOCK if failed rules are a strict majority of the pass/fail votes; ties PASS. (Accepted by the server; not yet selectable in the dashboard) |
| `weighted` | BLOCK when the **policy confidence ≥ threshold** |

Two short-circuits sit above every strategy: a
[block override](#validator-overrides) forces BLOCK, and an `error` on an
`is_decider` rule forces BLOCK.

### `any` — one bad rule is enough

The guardrail default. One `fail` anywhere → BLOCK. Per-rule thresholds are
the whole story.

### `all` — strict mode

Everything that ran must have **passed**. A `fail` blocks; an executed
`error` also blocks (it is not a pass) — making `all` the strategy that
fails *closed* on validator errors. Skips stay neutral: partial input does
not block.

### `majority` — vote

Only `pass` and `fail` rules vote. `fail` votes must be a **strict**
majority: 2 fails of 3 votes blocks; 1 of 2 (a tie) passes; 1 of 3 passes —
the same injection that blocks instantly under `any` can pass under
`majority` if the other rules disagree. Skips and errors don't vote.

### `weighted` — aggregated confidence

The dashboard's "Weighted score": *Block when the policy confidence ≥
threshold*. The math, exactly:

1. **Per rule** (only rules that produced a score): badness = `score` for
   `risk` polarity, `1 − score` for `quality` polarity.
2. **Per ruleset**: badness = mean over its scored rules.
3. **Policy confidence** = weighted mean of ruleset badness, using the
   ruleset weights from the editor (it keeps them summing to 100%;
   unset weights default to equal) — computed **only over rulesets that
   scored at least one rule**, with the weights renormalized over those.
4. `confidence >= weightedThreshold` → BLOCK. The comparison is `>=`:
   landing exactly on the line blocks.

Worked example — policy with two rulesets, threshold `0.5`:

| Ruleset | Weight | Rule outcome | Badness |
|---|---|---|---|
| Injection | 0.8 | `prompt-injection` fail, score 0.99 | 0.99 |
| Leakage | 0.2 | `data-leakage` pass, score 0.03 | 0.03 |

Confidence = `0.8×0.99 + 0.2×0.03` = **0.798** ≥ 0.5 → **BLOCK**, with
`aggregate_score: 0.798` in the envelope. Flip the weights (0.2/0.8) and the
same scores give `0.222` → **PASS** — the weights, not the per-rule
thresholds, decide.

**Renormalization under partial input.** Send only a prompt to that
flipped-weights policy (0.2 injection / 0.8 leakage) and the leakage ruleset
skips (`missing_input:llm_output`). Naive math would give
`0.2 × 0.99 = 0.198` → PASS — the skipped ruleset's weight silently
deflating the verdict, an easy dodge for hostile input. Instead the skipped
ruleset's weight is **renormalized away**: confidence is computed over what
ran (`0.99`) → **BLOCK**. Partial input is judged on the rules it actually
exercised, at their configured relative weights.

Note the corollary: under `weighted`, a rule can fail its *own* threshold
while the *policy* still passes (the aggregate stayed under the line), and
vice versa. Check `decision` — the per-rule statuses are evidence, not the
verdict.

**Defensive degradations** (each logged server-side): an unknown
`aggregation` value or a `weighted` policy with an invalid threshold
(`≤ 0` or `> 1`) falls back to `any`; contributing rulesets that all carry
weight 0 fall back to equal weights. `aggregation` in the envelope always
names the strategy that **actually** decided.

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

## Input coverage: all, some, or none

What happens when your input shape and the policy's rulesets don't fully
overlap? Exactly three cases:

| Your input covers… | What happens | Verdict driven by |
|---|---|---|
| **All rulesets** | Every rule executes | Real scores, full strategy semantics |
| **Some rulesets** | Matching rules execute; the rest record `skipped (missing_input:…)` | Only the rules that ran — skips are neutral, weights renormalize |
| **No rulesets** | Every rule skips | **PASS by vacuity**: nothing executed, nothing billed, `aggregate_score` absent |

The sharp edge is the last row: a PASS where **zero rules ran** is weaker
than it looks. `any_blocking` returns `False` (nothing blocked — true), but
nothing was evaluated either. For hard security gates, detect it and fail
closed:

```python
d = parse_policy(envelope)
ran = [r for rs in d.rulesets for r in rs.rules if r.status != "skipped"]
if not ran:
    # The policy never actually judged this input — the input shape doesn't
    # match any rule's required fields. Fix the request (see the discovery
    # endpoint's required_input_fields), or treat as unevaluated:
    raise PolicyNotApplicable(d.policy_name)
```

The Decisions ledger flags these too — the evidence drawer shows every rule
as SKIPPED with its reason, under a "no rules executed" banner.

---

## Inputs: one bag, every validator

A policy can span multiple validator domains. The request object's fields are
serialized once and sent to **every** validator the policy lists — each reads
the fields it needs:

| Request field | Wire field | Read by |
|---|---|---|
| `prompt` | `llm_input_query` | input / security / RAG validators — plus the output validators that compare against the question (`answer_relevance`, `conceptual_similarity`, `factual_consistency`, `intent_compliance`, `child_safety`, …) |
| `context` | `llm_input_context` | RAG / context-aware output validators |
| `response` | `llm_output` | output / RAG / security validators |
| `conversation_history`, `tool_calls`, `agent_responses`, `reference_data` | same names | agentic validators |

The "read by" column is indicative — the authoritative per-validator list is
the server's catalog, surfaced per policy as `required_input_fields` by the
[discovery endpoint](#discovering-policies-from-code).

If a validator's required field is missing, the server records that rule as
`status: "skipped"` with `skipped_reason: "missing_input:<fields>"` — skips
are **neutral** to the decision, never billed, and name exactly what to add.
Supply the union of what your policies need; the
[discovery endpoint](#discovering-policies-from-code) tells you the union
up front via `required_input_fields`.

Two serialization details that matter:

- **Empty means absent.** Fields left as `None` are omitted from the wire,
  and the server additionally treats empty / whitespace-only strings (and
  empty lists) as absent — `prompt=""` makes the matching rules skip with
  `missing_input:llm_input_query`, exactly like omitting it. A rule only
  runs against a field with actual content.
- **Extra fields are harmless.** Validators read only what they need — a bag
  carrying `prompt` + `response` + agentic fields satisfies an input policy,
  an output policy, and an agentic policy in the same call.

---

## Validator overrides

A policy can carry two override lists, keyed by **validator name** (set at
authoring time):

- **Allow overrides** — the named validators are exempted: their rules record
  `skipped (overrides_allow)`, neutral and unbilled.
- **Block overrides** — the named validators are force-failed **without
  running**: `status: "fail"`, `skipped_reason: "overrides_block"`,
  `has_score: false`. A forced fail blocks the policy under **every**
  strategy — it's the deny-overrides trump card, and no threshold change can
  allow it.

In the typed API a forced fail is a `PolicyRule` with `status == "fail"` and
`score is None` — render it as a verdict, not a number.

---

## Sync vs. async policies

A policy's **enforcement** (execution mode, set at authoring time) decides
what its envelope contains:

- **Sync** — validators run in-line; HTTP 200, `data.status: "completed"`,
  `decision` + `rulesets` present. This is the guardrail case.
- **Async** — the request is accepted (HTTP 202, `code: "DSQ-2020"`,
  `data.status: "accepted"`), evaluation runs in the background, and the
  verdict lands on the **Decisions** dashboard. The envelope carries **no
  decision, no rulesets, and no `credit_details`** (billing happens
  out-of-band when the background run completes) — `any_blocking` treats it
  as not blocking. Use `is_async(envelope)` to detect it, and the envelope's
  `request_id` to reconcile the eventual dashboard decision.

Mixing sync and async policies in one `policies=[...]` list is fine — each
envelope self-describes. Use async for monitoring/audit policies where you
want the ledger trail without paying the latency on the request path.

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
— build your request object from it and nothing will skip. It is computed
from the same server-side catalog that decides `missing_input` skips, so
what discovery promises is exactly what evaluation checks.

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

## Error handling

Two different failure planes — don't conflate them:

**Transport / HTTP errors** raise from the call:

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

**Validator errors inside a completed evaluation** do *not* raise — the call
returns HTTP 200 and the affected rules carry `status: "error"`. The policy
still decides: an error blocks if the rule is `is_decider` or the strategy
is `all`; otherwise the errored rule is neutral (under `weighted` it simply
doesn't contribute to the confidence). If every rule skipped or errored, the
decision is a vacuous PASS with no `aggregate_score` — the
[fail-closed pattern](#input-coverage-all-some-or-none) catches this case
too if you filter on `status == "pass" or status == "fail"` instead of just
excluding skips. Mark your load-bearing validators `is_decider` in the
policy editor if an ML outage must close the gate.

---

## Billing, latency, and publish propagation

- **Billing is per executed validator**, at the validator's pricing tier.
  Skipped rules cost nothing; forced fails (`overrides_block`) never call
  the ML service and cost nothing. Sync envelopes report the deduction in
  `credit_details` (omitted when nothing executed); async policies bill
  out-of-band on completion — the 202 carries no `credit_details`.
- **Latency**: each policy in `policies=[...]` is one sequential HTTP call
  from the SDK, and inside each call the policy's validators execute
  together server-side. The `timeout` you set on `Client` applies **per
  call** — budget end-to-end latency as roughly the sum over policies of
  the slowest validator in each. Keep request-path policies sync and small;
  push heavy audit bundles to async policies.
- **Publish propagation**: the server caches policy definitions briefly
  (with event-driven invalidation on publish). A newly published version is
  typically live within seconds; worst case, one cache TTL (~60s). The
  envelope's `policy_version` always tells you which version judged.

---

## The Decisions ledger

Every evaluation — sync or async, BLOCK or PASS — lands as one row on
**Dashboard → Realtime Policies → Decisions**, attributed to your
`application_name`. Click a row and the **Violation evidence** drawer shows
the decision the way the policy editor is structured:

- per-**ruleset** groups, each with its verdict chip (FAILED / PASSED /
  SKIPPED / ERROR);
- per-rule rows: executed rules with score-vs-threshold detail, skipped
  rules with their `missing_input:…` reason, forced fails as verdicts
  without fabricated scores;
- an explicit *"no rules executed — passed by default"* banner on vacuous
  decisions;
- for weighted policies, the tuning hints tell you which threshold change
  would have flipped a threshold-driven fail.

Use the ledger in reviews the way you'd use access logs: filter by
application, policy, or verdict; the `request_id` in your logs matches the
decision's evidence.

---

## Production patterns

### Policy ids are configuration

```python
POLICIES = [p for p in os.environ.get("DISSEQT_POLICIES", "").split(",") if p]

client = Client(
    project_id=..., api_key=...,
    application_name="checkout-bot",
    policies=POLICIES,        # [] → no default; populated → governed client
)

result = client.validate(req)                  # default applies
result = client.validate(req, policies=[...])  # per-call override when needed
```

Empty env → ungated code path; populated → every `validate()` on this client
is governed. No deploy to change which policies apply.

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

### Fail closed on unevaluated policies

For hard gates, a PASS with zero executed rules should not open the gate
(see [Input coverage](#input-coverage-all-some-or-none)):

```python
from disseqt_sdk import parse_policy

def gate(result) -> bool:
    for envelope in result["policies"]:
        d = parse_policy(envelope)
        if d is None or d.decision == "BLOCK":
            return False
        if d.enforcement == "sync" and not any(
            r.status in ("pass", "fail")
            for rs in d.rulesets for r in rs.rules
        ):
            return False          # vacuous PASS — nothing actually judged
    return True
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
| `application_name` | `None` | **Required** to evaluate policies (client default or per-call); shown on the Decisions ledger. |
| `policies` | `None` | Default policy-id list applied to **every** `validate()` call; per-call `policies=` overrides it. `[]` means "no default". Requires `application_name`. Copied defensively. |
| `realtime_policy_base_url` | `https://api.disseqt.ai/realtime-validations` | Base URL for the policy evaluate + discovery endpoints (served by production-monitoring next to the validators). Override for local testing. |
| `timeout` | `30` | Seconds, per HTTP call (each policy evaluation is one call). |

Public helpers: `any_blocking(result)` for the envelope; `is_blocking` /
`is_async` / `parse_policy` for individual policy envelopes — see
[the typed objects](#the-typed-objects) for the full field reference.
