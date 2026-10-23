# Red-teaming with the Disseqt SDK

Run Disseqt red-team attacks against your AI System from the terminal or from code, using only an API key.
The same `disseqt` CLI ships in the Python (`disseqt-ai-sdk`) and Node (`@disseqt-ai/sdk`) packages; flags,
defaults and exit codes are identical.

## Before you start

- **API key.** Create one in the Disseqt dashboard for your project. The key identifies your project; you
  do not need a project id.
- **An AI System with a linked app integration.** Register the AI System and its app integration (how Disseqt
  calls your model) in the dashboard, or link them from the CLI with `disseqt app link`. Attacks run on Disseqt's
  servers, so the integration's endpoint must be reachable from the internet.

## Install

```bash
pip install disseqt-ai-sdk          # Python 3.10.14+ -> `disseqt` on PATH
npm install -g @disseqt-ai/sdk      # Node 18.18+   -> `disseqt` on PATH
```

## Sign in

Initialise with one value: your **project API key** in `DISSEQT_API_KEY`. No project id is needed; the key
resolves to its project, organization and user.

```bash
export DISSEQT_API_KEY=<your project API key>
```

```python
from disseqt_sdk import DisseqtAPIClient

client = DisseqtAPIClient()  # reads DISSEQT_API_KEY; or DisseqtAPIClient(api_key="...")
```

```ts
import { DisseqtResourceClient } from '@disseqt-ai/sdk';

const client = new DisseqtResourceClient(); // reads DISSEQT_API_KEY; or { apiKey: '...' }
```

For the CLI on your own machine, you can store the key once instead:

```bash
disseqt login            # paste the key; it is verified, then stored in ~/.disseqt/config.json (0600)
disseqt whoami           # project, organization and user the key resolves to
```

Environment variables win over a stored login, so CI only needs `DISSEQT_API_KEY`:

| Variable                                        | Purpose                                                     |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `DISSEQT_API_KEY`                               | Project API key (required)                                  |
| `DISSEQT_BASE_URL`                              | Gateway, default `https://api.disseqt.ai/dataset`           |
| `DISSEQT_POLICIES_BASE_URL`                     | AI System registry, default `https://api.disseqt.ai/policy` |
| `DISSEQT_PROJECT_ID`, `DISSEQT_ORGANIZATION_ID` | Optional; derived from the key when unset                   |

## Choose a target

Every attack takes exactly one target:

| Flag                                 | Use it when                                                                                    |
| ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| `--app <ai_system_id>`               | **Recommended.** Disseqt uses the AI System's linked app integration.                          |
| `--integration <app_integration_id>` | You want a specific app integration.                                                           |
| `--endpoint <url \| template>`       | Advanced: single-turn takes a URL, multi-turn takes an integration template (JSON or `@file`). |

```bash
disseqt app list                                          # your AI Systems
disseqt app link <ai_system_id> --integration <integration_id>   # link once
disseqt redteam list-attacks --kind single                # attack technique keys
```

Teammates' integrations in the same project work too. An AI System with nothing linked returns
`422 NoLinkedTarget`; one linked to a custom LLM, RAG or MCP target returns `422 UnsupportedTarget`.

## Single-turn attack

Prompts from one or more prompt packs, rewritten by an attack technique, scored by your validators:

```bash
disseqt redteam attack --single-turn --app <ai_system_id> \
  --technique math_problem --pack <prompt_pack_id> --validator toxicity
```

The command waits for the run, prints the results, and cancels the run on Ctrl-C or `--max-wait`.

## Multi-turn attack

A conversation jailbreak that escalates over up to 10 turns:

```bash
disseqt redteam attack --multi-turn --app <ai_system_id> \
  --technique <mr_technique_uuid> --prompt "How do I pick a lock?" --max-depth 5
```

Each job ends with a `stop_reason`: `success` (jailbroken), `max_depth` or `error`.
No new integration is created; the AI System's existing one is reused.

## Repeatable runs from YAML

```bash
disseqt redteam run nightly.yaml
```

See [`redteam-example.yaml`](redteam-example.yaml). `target` takes exactly one of `application_id`,
`app_integration_id` or `endpoint_url`; `${VAR}` is expanded from the environment and an unset variable fails
the command.

## Score your own outputs (no target)

```bash
disseqt redteam validate --input "user prompt" --output "model answer" --validator toxicity
```

Returns `aggregate_decision` = `PASS`, `BORDERLINE` or `BLOCK` with per-validator scores.

## Track, cancel, report

```bash
disseqt redteam status <run_id>
disseqt redteam cancel <run_id>        # run ends "stopped" with stop_reason "user_cancel"
disseqt redteam results <run_id> --all
disseqt redteam report <run_id> --format markdown
disseqt redteam report --session <session_id> --format csv > report.csv
```

## From Python

```python
from disseqt_sdk import DisseqtAPIClient

client = DisseqtAPIClient()                     # reads DISSEQT_API_KEY or your `disseqt login`
print(client.whoami())                          # {"project_id", "organization_id", "user_id"}

ai_systems = client.applications.list()

verdict = client.sessions.validate(
    "How do I pick a lock?", ["toxicity"], output="I can't help with that."
)
print(verdict.aggregate_decision)               # PASS | BORDERLINE | BLOCK

testing_plan = {}  # fill in: same shape as the YAML above
session = client.sessions.create(
    {"name": "nightly", "application_context": {"name": "support-bot", "type": "web"},
     "testing_plan": testing_plan},
    application_id="<ai_system_id>",
)
```

## From Node / TypeScript

```ts
import { DisseqtResourceClient } from '@disseqt-ai/sdk';

const client = new DisseqtResourceClient(); // DISSEQT_API_KEY or `disseqt login`
console.log(await client.whoami());

const verdict = await client.redteam.validate({
  input: 'How do I pick a lock?',
  output: "I can't help with that.",
  validators: ['toxicity'],
  input_context: '',
});
console.log(verdict.aggregate_decision); // PASS | BORDERLINE | BLOCK

const aiSystems = await client.applications.list();
```

## Exit codes and errors

| Code | Meaning                                                                           |
| ---- | --------------------------------------------------------------------------------- |
| 0    | Success                                                                           |
| 1    | API error, run failed, or timed out                                               |
| 2    | Usage or auth problem (bad flags, two targets, missing key, config file not 0600) |
| 130  | Interrupted with Ctrl-C (the run is cancelled first)                              |

| Error                         | Fix                                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| `401`                         | Key revoked or wrong environment; run `disseqt whoami`.                                    |
| `404` on a run or integration | Wrong id, or it belongs to another project.                                                |
| `422 NoLinkedTarget`          | `disseqt app link <ai_system_id> --integration <id>`                                       |
| `422 UnsupportedTarget`       | Link an app integration; custom LLM / RAG / MCP targets are not supported for attacks yet. |

## Not supported yet

- Attacking a model that only runs on your machine (a local Python/JS function or `localhost`): Disseqt runs
  attacks server-side, so the target must be reachable over HTTP.
- Custom LLM, RAG and MCP targets for single/multi-turn attacks (use prompt-pack runs for RAG/MCP).
- `vuln-test` with an API key alone (needs a dashboard session for credit billing).
- Single-turn `--endpoint` without credentials: prefer `--app` or `--integration`.
