# Logging

The Disseqt SDK ships a built-in **structured logger** (`disseqt_logging`) used
by both the validation SDK (`disseqt_sdk`) and the agentic SDK
(`disseqt_agentic_sdk`). It has **no third-party dependencies** — it is built on
the Python standard library, so a plain `pip install disseqt-ai-sdk` gives you
the full logger with nothing else to install.

Highlights:

- **Silent by default** — emits nothing until you opt in, so upgrading never adds
  unsolicited output.
- **Structured** — JSON (for log pipelines) or a human console format (for local
  dev), with automatic `service` / `env` / `host` fields.
- **Safe** — automatic PII/credential redaction; request payloads are logged only
  as a content-free digest.
- **Familiar** — `logger.info(...)` / `debug` / `warning` / `error`, plus
  structured fields and `bind()`.

---

## Table of contents

- [Quick start](#quick-start)
- [Enabling and configuring](#enabling-and-configuring)
- [The log schema](#the-log-schema)
- [Logging from your own code](#logging-from-your-own-code)
- [What the SDK logs automatically](#what-the-sdk-logs-automatically)
- [PII & credential redaction](#pii--credential-redaction)
- [Writing to a file](#writing-to-a-file)
- [Backward compatibility](#backward-compatibility)
- [API reference](#api-reference)
- [Environment variables](#environment-variables)

---

## Quick start

The SDK is silent until you turn logging on. Enable it once at startup:

```python
import disseqt_sdk

disseqt_sdk.configure_logging(level="info")   # opt in

# ... now every client.validate(...) emits one structured line:
# {"event": "validation.response", "domain": "input-validation",
#  "slug": "toxicity", "status": 200, "latency_ms": 142.3,
#  "service": "disseqt-ai-sdk", "env": "production", "host": "...",
#  "logger": "disseqt_sdk.client", "timestamp": "2026-06-30T13:21:16.305Z"}
```

Or enable it without touching code, via an environment variable:

```bash
DISSEQT_LOG_LEVEL=info python my_app.py
```

That's it. If you never call `configure_logging()` / `set_log_level()` and never
set `DISSEQT_LOG_LEVEL`, the SDK produces no log output at all.

---

## Enabling and configuring

There are three equivalent ways to **enable** logging (any one opts in):

| How | Example |
| --- | --- |
| Programmatic | `disseqt_sdk.configure_logging(level="debug")` |
| Level setter | `disseqt_sdk.set_log_level("warning")` |
| Environment | `DISSEQT_LOG_LEVEL=info` |

### `configure_logging(...)`

`disseqt_sdk.configure_logging` (an alias of `disseqt_logging.configure`) accepts
keyword overrides:

```python
import sys
import disseqt_sdk

disseqt_sdk.configure_logging(
    level="debug",        # debug | info | warn | error
    fmt="json",           # json | console | auto (console on a TTY, else json)
    service="my-service", # bound to the "service" field on every line
    env="staging",        # bound to the "env" field
    redact=True,          # PII/credential redaction (default True)
    stream=sys.stderr,    # where to write (default stderr)
)
```

You can also pass a full config object:

```python
from disseqt_logging import LoggerConfig, configure

configure(LoggerConfig(level="info", service="my-service", env="prod", fmt="json"))
```

### Changing the level later

```python
import disseqt_sdk

disseqt_sdk.set_log_level("debug")   # also enables output if it was silent
```

### Silencing again

```python
import disseqt_logging

disseqt_logging.disable()   # no further output until re-enabled
```

### Defaults

| Setting | Default |
| --- | --- |
| Enabled? | **No** (silent until opted in) |
| Level (once enabled) | `info` |
| Format | `auto` — console when stderr is a TTY, JSON otherwise |
| Destination | `stderr` |
| Redaction | On |
| `service` | `disseqt-ai-sdk` |
| `env` | `production` |

---

## The log schema

Every JSON line carries a consistent set of keys:

| Key | Always present | Meaning |
| --- | --- | --- |
| `timestamp` | yes | ISO-8601 UTC, e.g. `2026-06-30T13:21:16.305Z` |
| `level` | yes | `debug` / `info` / `warning` / `error` |
| `event` | yes | the log message |
| `service` | yes | service name (configurable) |
| `env` | yes | environment (configurable) |
| `host` | yes | machine hostname |
| `logger` | yes | source module, e.g. `disseqt_sdk.client` |
| `version` / `component` | optional | set via config / `with_component()` |
| *(your fields)* | — | any structured fields you pass |
| `error` / `error_type` / `error_code` | on errors | the error envelope |
| `exception` | on errors with `exc_info` | the formatted traceback |

In **console** mode the same data renders as a single readable line:

```
2026-06-30T13:21:16.305Z [INFO] validation.response  domain=input-validation latency_ms=142.3 slug=toxicity status=200
```

---

## Logging from your own code

You can use the SDK's logger for your own application logs and get the same
structured output, redaction, and configuration.

```python
from disseqt_logging import get_logger

log = get_logger(__name__)

log.info("user signed in", user_id="u_123", plan="pro")
log.debug("cache lookup", key="profile:u_123", hit=True)
log.warning("rate limit near", remaining=5)
```

### Structured fields

Pass fields as keyword arguments — they become top-level JSON keys:

```python
log.info("order placed", order_id="o_42", amount_cents=19900, currency="USD")
# {"event": "order placed", "order_id": "o_42", "amount_cents": 19900, "currency": "USD", ...}
```

The stdlib `extra={...}` form also works (it is merged into the fields), so
existing call sites port over unchanged:

```python
log.info("order placed", extra={"order_id": "o_42"})
```

### Errors and exceptions

`error()` takes an optional exception and stamps an error envelope; pass
`exc_info=True` (inside an `except`) to include the traceback:

```python
try:
    charge(order)
except PaymentError as exc:
    log.error("charge failed", exc, order_id="o_42")
    # {"event": "charge failed", "error": "card declined",
    #  "error_type": "PaymentError", "order_id": "o_42", ...}

# or, capture the active exception + traceback:
try:
    charge(order)
except PaymentError:
    log.error("charge failed", exc_info=True, order_id="o_42")
    # adds "exception": "Traceback (most recent call last): ..."
```

If your exception object has a `.code` attribute it is emitted as `error_code`.

### Bound sub-loggers

`bind()` returns a logger that attaches the given fields to **every** line —
handy for request- or job-scoped context. `with_component()` is a shorthand that
sets the `component` field:

```python
req_log = get_logger(__name__).bind(request_id="r_abc", user_id="u_123")
req_log.info("handling request")     # both fields on every line
req_log.info("done", duration_ms=12)

worker = get_logger(__name__).with_component("ingest-worker")
worker.info("batch started", size=500)   # {"component": "ingest-worker", ...}
```

### Privacy-safe digests

To reference a large or sensitive payload without logging its contents, use
`digest()` — it emits a length plus a short SHA-256 prefix and is exempt from
redaction:

```python
from disseqt_logging import digest

log.info("prompt received", prompt_digest=digest(user_prompt))
# {"event": "prompt received", "prompt_digest": "len=215 sha256=fe51085aca29ddda", ...}
```

### Module-level shortcuts

For quick one-offs there are module-level functions backed by a default logger:

```python
import disseqt_logging as dl

dl.info("starting up", port=8080)
dl.error("boom", exc_info=True)
```

---

## What the SDK logs automatically

### Validation SDK (`disseqt_sdk`)

`Client.validate()` emits one event per call (once logging is enabled):

| Event | Level | Key fields |
| --- | --- | --- |
| `validation.request` | debug | `domain`, `slug`, `url`, `payload_digest`, `timeout_s` |
| `validation.response` | info | `domain`, `slug`, `status`, `latency_ms` |
| `validation.http_error` | error | `domain`, `slug`, `status`, `latency_ms`, `response_body_digest` |
| `validation.network_error` | error | `domain`, `slug`, `latency_ms` (+ traceback) |
| `validation.decode_error` | error | `domain`, `slug`, `status`, `latency_ms` (+ traceback) |

The request **payload is never logged verbatim** — only `payload_digest`. Auth
headers, `api_key`, and `project_id` are never passed to the logger.

To see the `validation.request` line (with the digest and URL), enable `debug`:

```python
disseqt_sdk.configure_logging(level="debug")
```

### Agentic SDK (`disseqt_agentic_sdk`)

The agentic SDK logs its lifecycle (client init, span buffering/flush, transport
results) through the same logger. Its public helpers are unchanged, and
`get_logger()` still returns a **standard-library `logging.Logger`** — so pass
structured fields via `extra={...}` (stdlib style):

```python
from disseqt_agentic_sdk import get_logger, set_log_level

set_log_level("debug")
log = get_logger(__name__)            # a real logging.Logger
log.info("span queued", extra={"span_count": 7})
```

---

## PII & credential redaction

Redaction is **on by default** and runs over every string field. Two strategies
are applied:

**1. Sensitive field names → `[REDACTED]`** (case-insensitive substring match):

`password`, `passwd`, `pwd`, `token`, `secret`, `api_key`, `apikey`, `api-key`,
`private_key`, `access_key`, `cookie`, `authorization`, `session`, `credential`,
`signature`, `prompt`, `csv`, `upload`, `project_id` (and `project-id`).

```python
log.info("auth", api_key="dsk_live_123", authorization="Bearer eyJ...")
# {"api_key": "[REDACTED]", "authorization": "[REDACTED]", ...}
```

**2. Sensitive value shapes** in any other string field:

| Shape | Replaced with |
| --- | --- |
| Email address | `[EMAIL]` |
| JWT (`eyJ….eyJ….…`) | `[JWT]` |
| Credit-card-like (13–19 digits) | `[CC]` |
| Phone-like (9+ digits) | `[PHONE]` |
| Long opaque token (32+ chars) | `[TOKEN]` |

```python
log.info("contact", note="reach me at jane@acme.com or 415-555-0134")
# {"note": "reach me at [EMAIL] or [PHONE]", ...}
```

> Non-string values (ints, floats, bools) and `digest()` outputs are never
> altered. Tracebacks are not scrubbed — avoid putting secrets in exception
> messages.

You can use the redaction helpers directly, or turn redaction off:

```python
from disseqt_logging import redact_string, sensitive_key

redact_string("token eyJa.eyJb.cccc")   # -> "token [JWT]"
sensitive_key("api_key")                 # -> True

# Disable (NOT recommended in production):
disseqt_sdk.configure_logging(level="info", redact=False)   # or DISSEQT_LOG_REDACT=0
```

---

## Writing to a file

Set a file path to also tee output to a size-rotated file (in addition to the
stream):

```python
from disseqt_logging import LoggerConfig, configure

configure(LoggerConfig(
    level="info",
    fmt="json",
    file_path="/var/log/myapp/disseqt.log",
    file_max_bytes=100 * 1024 * 1024,   # rotate at 100 MB
    file_backups=7,                     # keep 7 rotated files
))
```

Or via the environment: `DISSEQT_LOG_FILE=/var/log/myapp/disseqt.log`.

---

## Backward compatibility

- **No new dependencies** and **no install changes** — the logger is
  standard-library only.
- **Silent by default**, so existing installs see no new output after upgrading.
- The SDK uses a private root logger named `disseqt_ai_sdk` and does **not**
  propagate into your application's root logger, so it never captures or blocks
  loggers you own (including `disseqt.*` names).
- **`disseqt_agentic_sdk.get_logger()` still returns a stdlib `logging.Logger`**
  (unchanged type) — `isinstance(..., logging.Logger)`, `.setLevel(...)`,
  `.addHandler(...)`, `.handlers`, `.level`, etc. all continue to work. The only
  visible change is the output *format* (now structured) and that it is silent
  until enabled. Pass structured fields via `extra={...}`.

The validation/`disseqt_logging` `get_logger()` returns the richer
`disseqt_logging.Logger` wrapper (the `**fields` API shown above); use
`disseqt_logging.stdlib_logger(name)` if you specifically want a plain stdlib
logger there too.

---

## API reference

Importable from `disseqt_logging` (and the most common ones re-exported from
`disseqt_sdk` as noted):

| Symbol | Description |
| --- | --- |
| `get_logger(name=None) -> Logger` | Get the `**fields` wrapper logger (also `disseqt_sdk.get_logger`). |
| `stdlib_logger(name=None) -> logging.Logger` | Get a plain stdlib logger under the SDK root (use `extra={...}`). |
| `configure(config=None, **overrides) -> LoggerConfig` | Enable & configure (alias `disseqt_sdk.configure_logging`). |
| `set_level(level)` | Set level / enable (alias `disseqt_sdk.set_log_level`). |
| `current_level() -> str` | Current level name. |
| `disable()` | Silence the logger. |
| `digest(value) -> str-like` | Privacy-safe `len=… sha256=…` token. |
| `info/warn/error/debug(msg, **fields)` | Module-level default-logger emit. |
| `redact_string(s)` / `sensitive_key(name)` / `redact_field(name, value)` | Redaction helpers. |
| `parse_level(level)` / `level_name(int)` | Level name/number conversion. |
| `LoggerConfig` | Configuration dataclass (see below). |
| `Logger` | The logger class (returned by `get_logger`). |

**`Logger` methods:** `debug(msg, **f)`, `info(msg, **f)`, `warn(msg, **f)`,
`warning(msg, **f)`, `error(msg, err=None, **f)`, `bind(**f) -> Logger`,
`with_component(name) -> Logger`. All emit methods also accept `extra={...}` and
`exc_info=True|exc`.

**`LoggerConfig` fields:** `level`, `service`, `env`, `version`, `component`,
`fmt` (`json`/`console`/`auto`), `redact`, `stream`, `file_path`,
`file_max_bytes`, `file_backups`; plus `LoggerConfig.from_env(**overrides)`.

---

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DISSEQT_LOG_LEVEL` | _(unset → silent)_ | `debug`/`info`/`warn`/`error`; setting it **enables** output |
| `DISSEQT_LOG_FORMAT` | `auto` | `json`, `console`, or `auto` (console on a TTY) |
| `DISSEQT_LOG_SERVICE` | `disseqt-ai-sdk` | value bound to the `service` field |
| `DISSEQT_ENV` | `production` | value bound to the `env` field |
| `DISSEQT_LOG_VERSION` | — | value bound to the `version` field |
| `DISSEQT_LOG_REDACT` | `1` | set to `0` to disable redaction |
| `DISSEQT_LOG_FILE` | — | also tee output to this (rotated) file |

> Note: `configure_logging(...)` / `set_log_level(...)` arguments take precedence
> over environment variables for the values you pass.
