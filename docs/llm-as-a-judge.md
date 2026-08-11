# LLM-as-a-Judge

Grade any paired validator with a **certified LLM judge** running on your own
LLM account, instead of Disseqt's classic ML model.

The division of labor: **you bring the model** (OpenAI, Anthropic, Bedrock, or
any OpenAI-compatible endpoint — your key, your bill, your data agreement);
**Disseqt brings the rubric** — versioned, certified grading criteria, a fixed
scoring formula, and a verdict you can trace to the exact rubric version that
produced it.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Choosing which LLM judges](#choosing-which-llm-judges)
- [Reading the result](#reading-the-result)
- [Safety judges vs quality judges](#safety-judges-vs-quality-judges)
- [Custom labels](#custom-labels)
- [Running multiple judges in parallel](#running-multiple-judges-in-parallel)
- [Billing](#billing)
- [Errors](#errors)
- [FAQ](#faq)

## Prerequisites

1. **An LLM Integration in your project.** Dashboard → AI Inventory →
   LLM Integrations → *Add Custom LLM*. Add your provider key and run the
   connection test. Integrations are **per project, per environment** — one
   created in another project (or on staging vs production) is not visible
   here.
2. **Its Integration ID.** The table's **ID** column (and the row's view
   modal) has one-click copy. This UUID is what you pass as `llm_id`.
   - It is the **integration's** id, not a model name — the Model column
     already names the model; one integration pins one model.
   - Only **Permanent** integrations have a usable id. Temporary models are
     session-only; the backend has never seen their ids.

## Quickstart

```python
from disseqt_sdk import Client, SDKConfigInput
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.validators.input.safety import ToxicityValidator

client = Client(project_id="proj_123", api_key="dsk_...")

config = SDKConfigInput(
    threshold=0.5,
    llm_as_a_judge=True,
    llm_id="25dc0684-a394-4389-ad59-90b27138badf",  # copied from the dashboard
)

result = client.validate(
    ToxicityValidator(data=InputValidationRequest(prompt="have a lovely day"),
                      config=config)
)

print(result["score"])                                   # 0.0082
print(result["threshold_validated_result"])              # "Pass"
print(result["result"]["data"]["others"]["model"])       # which LLM judged
```

`llm_id` is **mandatory** with `llm_as_a_judge=True` and is validated at
construction — a missing id raises `ValueError` immediately, on your machine,
instead of a server error after the request is in flight. The mirror rule
also raises: `llm_id` without the flag would be a silent no-op (the classic
ML validator would run and ignore it), so the SDK refuses the combination.

## Choosing which LLM judges

Every judge run selects an integration **explicitly**. There is no implicit
"first one" or "most recent" — with several integrations in a project, the id
you pass is the whole answer.

```python
GPT5 = "25dc0684-..."   # integration pinned to gpt-5
MINI = "a65746a0-..."   # integration pinned to gpt-4o-mini

strict  = SDKConfigInput(threshold=0.5, llm_as_a_judge=True, llm_id=GPT5)
cheap   = SDKConfigInput(threshold=0.5, llm_as_a_judge=True, llm_id=MINI)
```

Per-call overrides, for power users, via the `judge` dict:

```python
config = SDKConfigInput(
    threshold=0.7,
    llm_as_a_judge=True,
    llm_id=GPT5,
    judge={
        "model": "gpt-5-mini",        # override the integration's model
        "criteria": "Penalize any answer that does not cite a source.",
    },
)
```

| Key | Effect |
|---|---|
| `model` | Overrides the integration's pinned model for this call |
| `criteria` | Extra grading guidance — **quality judges only** (see below) |
| `custom_llm_id` | Legacy spelling of the integration id; `llm_id` wins on conflict |

What you can **not** choose per call: the provider, endpoint, and API key
always come from the stored integration — they are server-authoritative, so a
request can never redirect your decrypted key to an arbitrary host.

## Reading the result

Never assume which model ran — read the receipts. Every judge verdict carries
them in `result["result"]["data"]["others"]`:

| Field | Meaning |
|---|---|
| `engine` | `"llm-judge"` — confirms a judge (not the ML model) produced this |
| `model` | The model that actually judged (e.g. `"gpt-5"`) |
| `rubric_version` | The certified rubric version (e.g. `"v17"`). Same rubric + same validator ⇒ same behavior, forever; changes ship as new versions |
| `reasoning` | The judge's written justification for the verdict |
| `severity` | Safety judges: the 1–10 severity the score derives from |
| `raw_label` | The judge's own label before display mapping |
| `scoring_path` | Quality judges: `"logprob_weighted"` or `"rating_fallback"` — which scoring formula ran (provider-dependent; see FAQ) |
| `criteria_ignored` | Present when you sent `criteria` to a certified safety judge (it ran its frozen rubric anyway) |
| `inference_time_ms` | Judge inference latency on your provider |

And at the **top level** of the response:

| Field | Meaning |
|---|---|
| `validator_name` | The validator that actually ran (`"llm-judge-toxicity"`) |
| `origin_validator` | What you asked for, when the judge reroute renamed the run (`"toxicity"`). Absent when you requested a judge slug directly |

## Safety judges vs quality judges

The two families score in **opposite directions** — misreading this inverts
every threshold decision:

| | Safety (toxicity, hate speech, violence, …) | Quality (helpfulness, coherence, relevance, …) |
|---|---|---|
| Score means | Higher = **worse** (severity) | Higher = **better** |
| Default threshold | 0.5 — run **fails at/above** it | 0.7 — run **passes at/above** it |
| How scored | Fixed curve over the judge's 1–10 severity | Continuous logprob-weighted rating (OpenAI) or the integer rating (other providers) |
| `criteria` | **Ignored** — the certified rubric runs verbatim; the response stamps `criteria_ignored` | Applied as additional grading guidance |

Certified safety rubrics being immune to caller text is a feature: it is what
lets a score mean the same thing across every caller and every run.

## Custom labels

`custom_labels` / `label_thresholds` work with judges exactly as with ML
validators, and are **display-only**: they re-bucket the score into your
naming, and never move the score, the pass/fail verdict, or the rubric.

Order labels along the judge's score axis — for a safety judge higher is
worse:

```python
config = SDKConfigInput(
    threshold=0.5,
    llm_as_a_judge=True,
    llm_id=GPT5,
    custom_labels=["OK", "Bad", "Awful", "Severe"],
    label_thresholds=[0.25, 0.5, 0.75],
)
```

## Running multiple judges in parallel

Selection is per-call and every request is independent down the whole chain,
so concurrent runs against different integrations are fully supported. The
`Client` is thread-safe to share (per-call HTTP, no shared session):

```python
from concurrent.futures import ThreadPoolExecutor

def judge(llm_id, text):
    cfg = SDKConfigInput(threshold=0.5, llm_as_a_judge=True, llm_id=llm_id)
    return client.validate(ToxicityValidator(
        data=InputValidationRequest(prompt=text), config=cfg))

with ThreadPoolExecutor(max_workers=8) as ex:
    gpt5_verdict = ex.submit(judge, GPT5, text)
    mini_verdict = ex.submit(judge, MINI, text)
```

Practical limits:

- Keep concurrency modest (≤ ~8): the platform caps concurrent judge
  dispatches, and sustained bursts beyond ~120 req/min per SDK key are rate
  limited.
- BYO means parallel judges share **your** provider quota — two integrations
  on the same OpenAI key share one TPM/RPM bucket. Watch your provider
  dashboard for throttling, not ours.
- **Do not compare raw scores across models as a shared scale.** Different
  models produce different score distributions, and quality judges may even
  use different scoring formulas per provider (`scoring_path`). Compare
  verdicts and disagreement rates; re-check tuned thresholds after switching
  judge models.

## Billing

- Each judge run costs **one credit**, charged by Disseqt — the LLM inference
  itself runs on your provider account and is billed by your provider.
- The credit is deducted **before** the judge is dispatched. Caller-fixable
  configuration errors (unknown integration, missing key) are rejected before
  billing and are free; failures after dispatch — provider auth errors,
  provider rate limits, timeouts — consume the credit.
- Judge responses do not include `credit_details`; check balances via the
  dashboard.

## Errors

| Error | When | Fix |
|---|---|---|
| `ValueError: llm_as_a_judge=True requires llm_id …` | At construction, before any request | Pass `llm_id` (copy from the dashboard ID column) |
| `ValueError: llm_id is only used with llm_as_a_judge=True …` | At construction | Set the flag, or drop `llm_id` |
| HTTP 400 | Integration not found / not in this project / no stored key | Verify the id belongs to **this** project and its key is Configured |
| HTTP 402 | Insufficient credits | Top up |
| HTTP 403 | Plan does not include the judge pack | Contact your admin |
| HTTP 429 | SDK rate limit | Back off; see parallelism limits above |
| HTTP 502 | The judge backend or **your provider** failed — including your provider rejecting the key or throttling you | Check the integration's connection test and your provider dashboard |

Note the last row: because inference runs on your account, a revoked key or
exhausted provider quota surfaces as a 502 from Disseqt. The connection test
on the integration and your provider's dashboard are the right diagnostic
tools.

## FAQ

**Which validators have judges?** Every paired validator reroutes when the
flag is set (e.g. `toxicity` → `llm-judge-toxicity`); the response records the
rename in `origin_validator`. A validator **without** a judge pairing falls
back gracefully to the classic ML validator — no error. Check
`others.engine` when you need certainty about which engine ran.

**Why is `llm_id` mandatory — doesn't the server have a default?** The server
supports a per-project default judge integration, but there is currently no
dashboard UI to set it, and an implicit default makes judge selection
unauditable. The SDK therefore requires the explicit id; if a default-judge
UI ships, this constraint can be relaxed.

**Why do the same inputs score slightly differently on OpenAI vs Anthropic or
Bedrock?** Quality judges use token logprobs when the provider exposes them
(OpenAI) and fall back to the judge's integer rating when it doesn't. The
result stamps which path ran in `others.scoring_path`. Safety judges use the
same severity formula everywhere, but the verdict-writing model still differs.

**Can I point a judge at my own hosted model?** Yes — create an integration
with the *Custom provider* (any OpenAI-compatible endpoint: vLLM, OpenRouter,
Groq, Together, LM Studio, …) and use its id as `llm_id`. HTTPS endpoints
only; internal/private addresses are rejected.

**Does my prompt/response data go to Disseqt's model providers?** No. Judge
inference goes to **your** configured provider under your own agreement.
Disseqt never uses its own keys on the judge path.
